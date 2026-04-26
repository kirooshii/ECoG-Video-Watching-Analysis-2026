"""
"ECoG Video Watching Analysis" by Horizons team
Horizons are:
Tobias Fleitas,
Makar Lavrov

DSP conventions
---------------
* All filters are zero-phase (filtfilt / MNE forward–backward) to avoid
  group-delay artefacts that would shift neural response latency.
* Notch filter Q=30 -> bandwidth ≈ powerline_freq/30 ≈ 1.67 Hz @50 Hz —
  narrow enough to preserve broadband signal, wide enough to kill harmonics.
* CAR (Common Average Reference) approximates a Laplacian on dense arrays
  and suppresses common-mode noise. Applied AFTER notch to avoid referencing
  noise back in.
* High-Gamma envelope: bandpass FIR (zero-phase, Kaiser window) -> |Hilbert|.
  The Hilbert approach preserves instantaneous amplitude modulation; a simple
  power-of-filtered-signal would introduce rectification artefacts.
* Epoching baseline: −0.2 s is long enough to estimate a pre-stimulus mean
  without aliasing the next trial's offset for stimuli presented ≥800 ms apart.
"""

from __future__ import annotations

import abc
import gc
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Tuple

import mne
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# silencing MNE's verbose channel-type warnings in pipeline contexts
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bci_pipeline")

# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class EpochConfig:
    """Shared epoching parameters for both modalities."""
    tmin: float = -0.2          # seconds before trigger (baseline window)
    tmax: float = 1.0           # seconds after trigger (analysis window)
    baseline: Tuple[float, float] = (-0.2, 0.0)   # baseline correction interval
    reject_peak_pv: Optional[float] = None         # µV threshold; None = no rejection
    chunk_size: int = 64        # epochs per RAM chunk during feature extraction


@dataclass
class ECoGConfig:
    """
    ECoG-specific DSP parameters — configured for Walk.mat clinical dataset.

    Walk.mat column layout (0-based Python indices):
      Col  0        : Time vector  -> discard, MNE owns the time axis
      Cols 1–160    : 160 ECoG electrodes  (n_ecog_channels)
      Col  161      : Photodiode  (binary 0/1, rising edge = stimulus onset)
      Col  162      : StimCode    (1=pre-paradigm, 2=video playing, 3=post)
      Col  163      : GroupId     -> discard
    """
    sfreq_expected:    float = 1200.0   # Hz — Walk.mat acquisition rate

    # column indices (0-based)
    mat_variable:      str   = "y"
    n_ecog_channels:   int   = 160      # total electrodes in file
    ecog_col_start:    int   = 1        # kept for reference, ROI slicing overrides
    ecog_col_stop:     int   = 161
    photodiode_col:    int   = 161
    stimcode_col:      int   = 162
    stimcode_video:    int   = 2
    photodiode_thresh: float = 0.5

    # ROI: visual / temporal / occipital channels (1-based, from electrode map)
    # channels 1–60 -> dense frontal/motor grid (excluded to avoid muscle artefacts and non-visual responses)
    # channels 61–100 -> posterior temporal, lateral occipital (lateral view)
    # channels 101–160 -> inferior temporal, fusiform, ventral occipital
    roi_channels: list = field(default_factory=lambda: list(range(61, 161)))

    # DSP parameters
    notch_freqs: list = field(default_factory=lambda: [50.0, 100.0, 150.0])
    notch_q: float = 30.0
    hg_band: Tuple[float, float] = (70.0, 150.0)
    fir_n_taps: int = 257


@dataclass
class UnicornConfig:
    """Unicorn Hybrid Black–specific DSP parameters."""
    sfreq: float = 250.0             # Hz — fixed hardware rate
    channel_names: list = field(default_factory=lambda: [
        "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"
    ])
    # Alpha (8–13 Hz) + Beta (13–30 Hz) = 8–30 Hz combined band.
    # High-Gamma (>70 Hz) is heavily attenuated by the skull (≈40 dB/decade),
    # dura, CSF, and scalp, so it is physiologically inaccessible at the scalp.
    ab_band: Tuple[float, float] = (1.0, 50.0)
    butter_order: int = 4 # order of the Butterworth bandpass filter
    trigger_column: str = "Trigger"  # column name in the Unicorn CSV export


# =============================================================================
# Abstract base class
# Definition of the contract every processor must fulfil
# =============================================================================

class SignalProcessor(abc.ABC):
    """
    Abstract base for a single-modality BCI preprocessing pipeline.

    Concrete subclasses implement:
      * load_raw()      -> mne.io.BaseRaw  (lazy, preload=False)
      * preprocess()    -> mne.io.BaseRaw  (filtered, referenced, still lazy)
      * extract_epochs()-> Generator yielding (X_chunk, y_chunk) pairs
    """

    def __init__(self, epoch_cfg: EpochConfig):
        self.epoch_cfg = epoch_cfg
        self._raw: Optional[mne.io.BaseRaw] = None

    @abc.abstractmethod
    def load_raw(self, path: Path) -> mne.io.BaseRaw:
        """
        Memory-map the raw file without pulling it into RAM.
        Must return an MNE Raw object with preload=False.
        """

    @abc.abstractmethod
    def preprocess(self) -> mne.io.BaseRaw:
        """Apply modality-specific filtering and referencing in-place."""

    @abc.abstractmethod
    def extract_epochs(self) -> mne.Epochs:
        """
        Segment the continuous signal around triggers.
        Returns an MNE Epochs object (also lazy until iterated).
        """

    def run(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orchestrates the full pipeline for one recording file.

        Returns
        -------
        X : np.ndarray, shape (n_epochs, n_channels)
        y : np.ndarray, shape (n_epochs,)

        Side-effect
        -----------
        Sets ``self.ecog_flash_seconds`` — absolute flash onset times in seconds
        at 1200 Hz resolution, used to synchronise the Unicorn multi-file pipeline.
        """
        log.info("[%s] Loading: %s", self.__class__.__name__, path.name)
        self._raw = self.load_raw(path)

        log.info("[%s] Preprocessing …", self.__class__.__name__)
        self._raw = self.preprocess()

        log.info("[%s] Extracting epochs …", self.__class__.__name__)
        epochs = self.extract_epochs()

        # Expose flash times for cross-modality sync
        self.ecog_flash_seconds: np.ndarray = (
            epochs.events[:, 0].astype(np.float64) / self.cfg.sfreq_expected
        )
        log.info(
            "[%s] Flash times stored: %d events, %.2f – %.2f s",
            self.__class__.__name__,
            len(self.ecog_flash_seconds),
            self.ecog_flash_seconds[0],
            self.ecog_flash_seconds[-1],
        )

        log.info("[%s] Building feature matrix …", self.__class__.__name__)
        X, y = self._build_features(epochs)

        del epochs
        del self._raw
        self._raw = None
        gc.collect()

        log.info("[%s] Done — X: %s  y: %s", self.__class__.__name__, X.shape, y.shape)
        return X, y

    def _build_features(
        self, epochs: mne.Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        iterating over epochs in fixed-size chunks to avoid materialising the
        full (n_epochs × n_channels × n_times) tensor.

        mean squared amplitude over the epoch window per
        channel = mean instantaneous power (band-limited by preprocessing).

        each chunk contributes rows to X of shape (chunk, C);
        stacking yields (n_epochs, n_channels).
        """

        epochs.drop_bad(verbose=False)

        X_parts, y_parts = [], []
        chunk = self.epoch_cfg.chunk_size
        n_total = len(epochs)

        for start in range(0, n_total, chunk):
            stop = min(start + chunk, n_total)

            data_chunk = epochs[start:stop].get_data(copy=False)  # (B, C, T)

            # Mean power = mean(signal squared) over time axis -> (B, C)
            power = np.mean(data_chunk ** 2, axis=-1)
            X_parts.append(power)
            y_parts.append(epochs[start:stop].events[:, 2])  # event id column


            # realeasing the chunks from memory to save RAM
            del data_chunk
            gc.collect()

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        return X, y


# =============================================================================
# ECoG Processor
# =============================================================================

class ECoGProcessor(SignalProcessor):
    """
    Pipeline order:
      load  notch -> CAR -> HG bandpass -> Hilbert envelope -> epoch

    CAR before bandpass!!
    
    referencing in broadband prevents the HG bandpass
    from creating reference-channel–specific spectral artefacts.
    """

    def __init__(self, epoch_cfg: EpochConfig, ecog_cfg: ECoGConfig):
        super().__init__(epoch_cfg)
        self.cfg = ecog_cfg

    # loading
    def load_raw(self, path: Path) -> mne.io.BaseRaw:

        cfg = self.cfg
        sfreq = cfg.sfreq_expected

        if path.suffix.lower() != ".mat":
            raise ValueError(f"ECoGProcessor expects a .mat file, got: {path.suffix}")
        

        # loading only y variable
        log.info("  scipy.io.loadmat('%s', variable_names=['%s']) …", path.name, cfg.mat_variable)
        mat = sio.loadmat(str(path), variable_names=[cfg.mat_variable])
        y = mat[cfg.mat_variable]
        
        
        if y.ndim == 2 and y.shape[0] < y.shape[1]:
            y = y.T

        log.info("  Raw matrix shape: %s  dtype: %s", y.shape, y.dtype)
        n_samples = y.shape[0]

        # slicing ROI channels and downcasting
  
        roi_cols = cfg.roi_channels                          # [61, 62, etc., 160]
        n_roi    = len(roi_cols)

        log.info("  Slicing %d ROI channels %d–%d, Photodiode [%d], StimCode [%d] …",
                 n_roi, roi_cols[0], roi_cols[-1],
                 cfg.photodiode_col, cfg.stimcode_col)

        ecog_data  = y[:, roi_cols].T.astype(np.float32)    # (n_roi, n_samples)
        photodiode = y[:, cfg.photodiode_col].astype(np.float32)
        stimcode   = y[:, cfg.stimcode_col  ].astype(np.float32)

        # freeing the original matrix to save RAM
        del y, mat
        gc.collect()
        log.info("  Original matrix freed.  ROI ECoG array: %s  (%.1f MB)",
                 ecog_data.shape, ecog_data.nbytes / 1e6)

        # building MNE RawArray for ROI channels only
        ch_names = [f"ECoG{ch:03d}" for ch in roi_cols]     # e.g. ECoG061…ECoG160
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=["ecog"] * n_roi,
        )
        raw = mne.io.RawArray(ecog_data * 1e-6, info, verbose=False)
        del ecog_data
        gc.collect()

        # Photodiode + StimCode
        aux_data = np.vstack([photodiode[np.newaxis, :], stimcode[np.newaxis, :]])
        aux_info = mne.create_info(
            ch_names=["Photodiode", "StimCode"],
            sfreq=sfreq,
            ch_types=["stim", "stim"],
        )
        aux_raw = mne.io.RawArray(aux_data, aux_info, verbose=False)
        raw.add_channels([aux_raw], force_update_info=True)

        del photodiode, stimcode, aux_data
        gc.collect()

        log.info("  RawArray ready: %d ECoG ch + Photodiode + StimCode  |  %.1f s @ %.0f Hz",
                 cfg.n_ecog_channels, n_samples / sfreq, sfreq)
        return raw

    # preprocessing
    def preprocess(self) -> mne.io.BaseRaw:

        raw = self._raw
        ecog_picks = mne.pick_types(raw.info, ecog=True)

        # Notch filter 
        log.info("  Applying notch filters at: %s Hz", self.cfg.notch_freqs)
        for freq in self.cfg.notch_freqs:
            b_notch, a_notch = sig.iirnotch(
                w0=freq,
                Q=self.cfg.notch_q,
                fs=raw.info["sfreq"],
            )

            for ch_idx in ecog_picks:
                ch_data, _ = raw[ch_idx, :]          # (1, T)
                filtered = sig.filtfilt(b_notch, a_notch, ch_data[0])
                raw._data[ch_idx, :] = filtered.astype(np.float32)
            gc.collect()

        # CAR
        # subtracts the instantaneous mean across all ECoG electrodes from each electrode
        log.info("  Applying Common Average Reference (CAR) …")
        ecog_data = raw._data[ecog_picks, :] 
        car_mean = ecog_data.mean(axis=0, keepdims=True)   # (1, T)
        raw._data[ecog_picks, :] -= car_mean 
        del car_mean
        gc.collect()

        # High-Gamma bandpass + Hilbert envelope
        log.info("  Computing High-Gamma (%.0f–%.0f Hz) envelope …", *self.cfg.hg_band)
        sfreq = raw.info["sfreq"]
        nyq = sfreq / 2.0
        low, high = self.cfg.hg_band[0] / nyq, self.cfg.hg_band[1] / nyq
        b_hg = sig.firwin(
            numtaps=self.cfg.fir_n_taps,
            cutoff=[low, high],
            pass_zero=False,          # bandpass
            window=("kaiser", 8.6),
            fs=2.0,                   # normalised frequency
        )

        for ch_idx in ecog_picks:
            ch_data = raw._data[ch_idx, :]                    # view (T,)
            filtered = sig.filtfilt(b_hg, [1.0], ch_data)    # zero-phase BPF
            # Hilbert transform
            envelope = np.abs(sig.hilbert(filtered)).astype(np.float32)
            raw._data[ch_idx, :] = envelope
            del filtered, envelope

        gc.collect()
        return raw

    # epoching 
    def extract_epochs(self) -> mne.Epochs:
        cfg = self.cfg
        raw = self._raw

        photodiode = raw["Photodiode"][0][0]   # (T,) float32
        stimcode   = raw["StimCode"  ][0][0]   # (T,) float32

        # detecting rising edges on the photodiode channel 
        thresh   = cfg.photodiode_thresh
        is_high  = photodiode >= thresh                     # (T,) bool
        rising   = is_high[1:] & ~is_high[:-1]             # (T-1,) bool
        all_rising_samples = np.where(rising)[0] + 1       # add +1 because is_high[1:] corresponds to original index 1..T-1.

        log.info("  Photodiode: %d total rising edges detected", len(all_rising_samples))

        # keeping only events where StimCode == 2 
        stim_at_onset = stimcode[all_rising_samples].astype(np.int32)
        video_mask    = stim_at_onset == cfg.stimcode_video
        valid_samples = all_rising_samples[video_mask]

        n_total = len(all_rising_samples)
        n_valid = len(valid_samples)
        n_dropped = n_total - n_valid
        log.info("  StimCode gate (==2): kept %d / %d  (dropped %d outside video)",
                 n_valid, n_total, n_dropped)

        del photodiode, stimcode, is_high, rising, all_rising_samples
        gc.collect()

        if n_valid == 0:
            raise ValueError(
                "No photodiode events found during StimCode==2 (video playing). "
                "Check that the Photodiode column index and threshold are correct, "
                f"and that StimCode value {cfg.stimcode_video} is present in the data."
            )


        # we've wateched the video and wrote down every event that happened during a white signal in the left down corner
        # we got 126 events: color, shape or face. this array represent them sequentially
        # array is used to train a model that we'll later use for analyzing Walk.mat and Unicorn data that we've got
        real_labels = np.array([
            1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1,
            3, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1,
            2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1,
            1, 1, 2, 2, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2,
            2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2,
            2, 2, 3, 3, 3, 3,
        ], dtype=np.int32)

        if len(real_labels) != n_valid:
            raise ValueError(
                f"Label count mismatch: you provided {len(real_labels)} labels "
                f"but {n_valid} valid photodiode events were detected. "
                "Check the StimCode gate or your label list."
            )
        labels = real_labels
        log.info("  Using real behavioural labels (%d events, classes: %s)",
                 n_valid, np.unique(labels).tolist())
      
        #  MNE events array
        # col0=sample_index, col1=prev_event_id (0), col2=event_id
        events = np.column_stack([
            valid_samples.astype(np.int32),
            np.zeros(n_valid, dtype=np.int32), # always zero
            labels,
        ])

        unique_codes = np.unique(labels)
        placeholder_names = {1: "color", 2: "shape", 3: "face"}
        event_id = {placeholder_names.get(c, f"stim_{c}"): int(c) for c in unique_codes}
        log.info("  Event IDs: %s", event_id)

        # epoch around each event
        reject_threshold = (
            {"ecog": self.epoch_cfg.reject_peak_pv * 1e-6}
            if self.epoch_cfg.reject_peak_pv is not None
            else None
        )
        if reject_threshold is not None:
            log.info(
                "  Artifact rejection enabled: peak-to-peak threshold = %.1f µV",
                self.epoch_cfg.reject_peak_pv,
            )
        else:
            log.warning(
                "  Artifact rejection DISABLED (EpochConfig.reject_peak_pv is None). "
                "Consider setting a peak-to-peak threshold (e.g. 500 µV) to exclude "
                "electrode pops and seizure artefacts from the feature matrix."
            )

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=self.epoch_cfg.tmin,
            tmax=self.epoch_cfg.tmax,
            baseline=self.epoch_cfg.baseline,
            picks=mne.pick_types(raw.info, ecog=True),
            preload=False, 
            reject=reject_threshold,
            reject_by_annotation=True,
            verbose=False,
        )
        return epochs


# =============================================================================
# Unicorn Hybrid Black Processor
# =============================================================================

class UnicornProcessor(SignalProcessor):
    """
    pipeline order per file)
      load -> bandpass (Alpha+Beta: 8–30 Hz) -> epoch from ECoG flash times -> features
    """

    def __init__(self, epoch_cfg: EpochConfig, unicorn_cfg: UnicornConfig):
        super().__init__(epoch_cfg)
        self.cfg = unicorn_cfg

    
    # loading
    def load_raw(self, path: Path) -> mne.io.BaseRaw:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv(path)
        elif suffix in (".xdf", ".lsl"):
            return self._load_lsl(path)
        else:
            raise ValueError(
                f"UnicornProcessor: unsupported format '{suffix}'. Use .csv or .xdf"
            )

    def _load_csv(self, path: Path) -> mne.io.BaseRaw:
        import csv

        log.info("  Reading CSV header …")
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        eeg_cols    = [h for h in header if h.strip() not in (self.cfg.trigger_column, "")]
        trig_col_idx = (
            header.index(self.cfg.trigger_column)
            if self.cfg.trigger_column in header
            else -1
        )

        log.info("  Loading CSV data (streaming via numpy) …")
        raw_csv = np.loadtxt(
            path, delimiter=",", skiprows=1, dtype=np.float32
        )   # (n_samples, n_cols)

        eeg_indices = [header.index(c) for c in eeg_cols]
        n_ch     = len(self.cfg.channel_names)
        eeg_data = raw_csv[:, eeg_indices[:n_ch]].T * 1e-6
        log.info(
            "  CSV has %d non-trigger columns; using first %d as EEG (%s …)",
            len(eeg_indices), n_ch, eeg_cols[n_ch - 1],
        )


        ch_names = self.cfg.channel_names[: len(eeg_cols)]
        info = mne.create_info(
            ch_names=ch_names, sfreq=self.cfg.sfreq, ch_types=["eeg"] * len(ch_names)
        )
        raw = mne.io.RawArray(eeg_data.astype(np.float32), info, verbose=False)
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

        if trig_col_idx != -1:
            trig_data = raw_csv[:, trig_col_idx][np.newaxis, :].astype(np.float32)
            trig_info = mne.create_info(["STI014"], self.cfg.sfreq, ch_types=["stim"])
            raw.add_channels(
                [mne.io.RawArray(trig_data, trig_info, verbose=False)],
                force_update_info=True,
            )

        del raw_csv, eeg_data
        gc.collect()
        return raw

    def _load_lsl(self, path: Path) -> mne.io.BaseRaw:
        """Load an XDF file recorded via Lab Streaming Layer."""
        try:
            import pyxdf
        except ImportError as exc:
            raise ImportError("Install pyxdf: pip install pyxdf") from exc

        streams, _ = pyxdf.load_xdf(str(path))
        eeg_stream = next(
            (s for s in streams if int(s["info"]["channel_count"][0]) == 8), None
        )
        if eeg_stream is None:
            raise ValueError("No 8-channel EEG stream found in XDF file.")

        data  = np.array(eeg_stream["time_series"]).T.astype(np.float32) * 1e-6
        sfreq = float(eeg_stream["info"]["nominal_srate"][0])
        info  = mne.create_info(
            ch_names=self.cfg.channel_names, sfreq=sfreq, ch_types=["eeg"] * 8
        )
        return mne.io.RawArray(data, info, verbose=False)

    # preprocessing
    def preprocess(self) -> mne.io.BaseRaw:

        sfreq     = self._raw.info["sfreq"]
        low, high = self.cfg.ab_band
        nyq       = sfreq / 2.0

        log.info(
            "  Bandpass filter: %.1f–%.1f Hz (Butterworth order %d) …",
            low, high, self.cfg.butter_order,
        )
        b, a = sig.butter(
            N=self.cfg.butter_order,
            Wn=[low / nyq, high / nyq],
            btype="bandpass",
        )
        eeg_picks = mne.pick_types(self._raw.info, eeg=True)
        for ch_idx in eeg_picks:
            ch_data = self._raw._data[ch_idx, :]
            self._raw._data[ch_idx, :] = sig.filtfilt(b, a, ch_data).astype(np.float32)

        gc.collect()
        return self._raw

    # epoch from pre-computed flash times
    def _epoch_from_flash_seconds(
        self,
        raw: mne.io.BaseRaw,
        flash_seconds: np.ndarray,
        labels: np.ndarray,
        label_map: dict,
    ) -> mne.Epochs:
        """
        raw           : filtered Unicorn RawArray
        flash_seconds : (n_trials,) onset times in seconds
        labels        : (n_trials,) integer class codes  (1=color, 2=shape, 3=face)
        label_map     : {int: str} for MNE event_id  (e.g. {1:"color", 2:"shape",...})
        """
        sfreq      = raw.info["sfreq"]
        n_samples  = raw.n_times

        flash_samples = np.round(flash_seconds * sfreq).astype(np.int32)

        # discard any events that fall outside the recording
        tmin_samp = int(np.round(self.epoch_cfg.tmin * sfreq))   # negative offset
        tmax_samp = int(np.round(self.epoch_cfg.tmax * sfreq))

        valid_mask = (
            (flash_samples + tmin_samp >= 0) &
            (flash_samples + tmax_samp <  n_samples)
        )
        n_dropped = (~valid_mask).sum()
        if n_dropped:
            log.warning(
                "  Dropped %d events outside recording bounds "
                "(offset may push early flashes before t=0 or beyond EOF).",
                n_dropped,
            )

        flash_samples = flash_samples[valid_mask]
        labels_valid  = labels[valid_mask]

        # MNE events: (n_events, 3) — [sample_idx, 0, event_id]
        events = np.column_stack([
            flash_samples,
            np.zeros(len(flash_samples), dtype=np.int32),
            labels_valid.astype(np.int32),
        ])

        event_id = {label_map[c]: int(c) for c in np.unique(labels_valid) if c in label_map}
        log.info("  Built %d epochs from flash times  |  event_id: %s", len(events), event_id)

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=self.epoch_cfg.tmin,
            tmax=self.epoch_cfg.tmax,
            baseline=self.epoch_cfg.baseline,
            picks=mne.pick_types(raw.info, eeg=True),
            preload=False, 
            verbose=False,
        )
        return epochs


    def run(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy single-file run — kept for backward compatibility."""
        log.info("[UnicornProcessor] Loading: %s", path.name)
        self._raw = self.load_raw(path)
        self._raw = self.preprocess()
        epochs    = self.extract_epochs() 
        X, y      = self._build_features(epochs)
        del epochs, self._raw
        self._raw = None
        gc.collect()
        return X, y


    def run_multi_file(
        self,
        unicorn_files:       list,           # [(Path, offset_seconds), ...]
        ecog_flash_seconds:  np.ndarray,     # (126,) from ECoGProcessor.ecog_flash_seconds
        real_labels:         np.ndarray,     # (126,) integer codes 1/2/3
        label_map:           dict,           # {1: "color", 2: "shape", 3: "face"}
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        unicorn_files       : list of (Path, float), file path + video-start offset
        ecog_flash_seconds  : absolute flash times from ECoG
        real_labels         : (n_trials,) — same 126 labels used for ECoG
        label_map           : {int: str} — class code

        Returns
        X_total : (n_files × n_trials, 8)
        y_total : (n_files × n_trials,)
        """
        X_parts, y_parts = [], []

        for file_idx, (csv_path, offset) in enumerate(unicorn_files, start=1):
            csv_path = Path(csv_path)
            log.info(
                "[UnicornProcessor] File %d/%d — %s  (offset=%.1f s)",
                file_idx, len(unicorn_files), csv_path.name, offset,
            )

            # load
            self._raw = self.load_raw(csv_path)

            # bandpass
            self._raw = self.preprocess()

            unicorn_flash_seconds = ecog_flash_seconds + offset
            log.info(
                "  Flash window in Unicorn time: %.2f – %.2f s",
                unicorn_flash_seconds[0], unicorn_flash_seconds[-1],
            )

            # epoch
            epochs = self._epoch_from_flash_seconds(
                self._raw, unicorn_flash_seconds, real_labels, label_map
            )

            log.info("  Extracting features (chunk_size=%d) …", self.epoch_cfg.chunk_size)
            X_file, y_file = self._build_features(epochs)
            log.info("  File features: X=%s  y=%s", X_file.shape, y_file.shape)

            X_parts.append(X_file)
            y_parts.append(y_file)

            del epochs, self._raw, unicorn_flash_seconds
            self._raw = None
            gc.collect()

        X_total = np.concatenate(X_parts, axis=0)   # (n_files × 126, 8)
        y_total = np.concatenate(y_parts, axis=0)   # (n_files × 126,)
        del X_parts, y_parts
        gc.collect()

        log.info(
            "[UnicornProcessor] Multi-file complete — X_total: %s  y_total: %s  "
            "classes: %s",
            X_total.shape, y_total.shape, np.unique(y_total).tolist(),
        )
        return X_total, y_total

    def extract_epochs(self) -> mne.Epochs:
        """Legacy trigger-channel epoching — used only by single-file ``run()``."""
        try:
            events = mne.find_events(self._raw, stim_channel="STI014", verbose=False)
        except ValueError:
            events, _ = mne.events_from_annotations(self._raw, verbose=False)

        if events.shape[0] == 0:
            raise ValueError("No events found — check trigger column or annotations.")

        unique_codes = np.unique(events[:, 2])
        event_id     = {f"stim_{c}": int(c) for c in unique_codes}

        return mne.Epochs(
            self._raw,
            events=events,
            event_id=event_id,
            tmin=self.epoch_cfg.tmin,
            tmax=self.epoch_cfg.tmax,
            baseline=self.epoch_cfg.baseline,
            picks=mne.pick_types(self._raw.info, eeg=True),
            preload=False,
            verbose=False,
        )

# =============================================================================
# Machine Learning Part
# =============================================================================

class BCIClassifier:

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.le = LabelEncoder()
        self._pipelines = self._build_pipelines()

    def _build_pipelines(self) -> dict[str, Pipeline]:
        return {
            "RandomForest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",   # to compensate lack of "face" in the training data from video
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )),
            ]),
            "SVM_RBF": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=self.random_state,
                    decision_function_shape="ovr",
                )),
            ]),
        }

    def fit_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: Optional[dict] = None,
    ) -> dict:
        """
        Parameters
        X : (n_epochs, n_features)
        y : (n_epochs,) — integer class labels
        label_names : {int: str} mapping for confusion matrix display

        Returns
        results : dict keyed by classifier name with metrics sub-dicts
        """
        y_enc = self.le.fit_transform(y)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        results = {}

        for name, pipe in self._pipelines.items():
            log.info("  Cross-validating: %s (%d folds) …", name, self.n_splits)
            cv_out = cross_validate(
                pipe,
                X, y_enc,
                cv=skf,
                scoring=scoring,
                return_train_score=False,
                n_jobs=self.n_jobs,
            )
            # final fit on all data for the confusion matrix
            pipe.fit(X, y_enc)
            y_pred = pipe.predict(X)

            cm = confusion_matrix(y_enc, y_pred)
            target_names = (
                [label_names[c] for c in self.le.classes_]
                if label_names
                else [str(c) for c in self.le.classes_]
            )

            results[name] = {
                "accuracy_mean":   cv_out["test_accuracy"].mean(),
                "accuracy_std":    cv_out["test_accuracy"].std(),
                "precision_macro": cv_out["test_precision_macro"].mean(),
                "recall_macro":    cv_out["test_recall_macro"].mean(),
                "f1_macro":        cv_out["test_f1_macro"].mean(),
                "confusion_matrix": cm,
                "target_names":    target_names,
                "report":          classification_report(y_enc, y_pred, target_names=target_names),
            }

        return results

    @staticmethod
    def print_results(results: dict, modality_label: str = "") -> None:
        """Pretty-print cross-validation results to stdout."""
        header = f"{'='*60}\n  {modality_label} — Classification Results\n{'='*60}"
        print(header)
        for clf_name, metrics in results.items():
            print(f"\n── {clf_name} ──────────────────────────────")
            print(f"  CV Accuracy : {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
            print(f"  Precision   : {metrics['precision_macro']:.4f}")
            print(f"  Recall      : {metrics['recall_macro']:.4f}")
            print(f"  F1 (macro)  : {metrics['f1_macro']:.4f}")
            print(f"\n  Classification Report (full-data fit):\n{metrics['report']}")
            print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")


# =============================================================================
#  BCI Experiment
# =============================================================================

class BCIExperiment:

    def __init__(
        self,
        epoch_cfg: EpochConfig,
        ecog_cfg: ECoGConfig,
        unicorn_cfg: UnicornConfig,
        n_cv_splits: int = 5,
    ):
        self.ecog_proc = ECoGProcessor(epoch_cfg, ecog_cfg)
        self.unicorn_proc = UnicornProcessor(epoch_cfg, unicorn_cfg)
        self.classifier = BCIClassifier(n_splits=n_cv_splits)

    def run(
        self,
        ecog_mat_path:    Optional[Path] = None,
        unicorn_files:    Optional[list] = None,   # [(Path, offset_s), ...]
        label_map:        Optional[dict] = None,
    ) -> dict:
        """
        Execute both pipelines and return a dict with all metrics.

        Parameters
        ecog_mat_path : Path to Walk.mat
        unicorn_files : list of (Path, float) tuples — CSV path + video-start offset
        label_map     : {int: str}  e.g. {1: "color", 2: "shape", 3: "face"}
        """
        all_results      = {}
        ecog_flash_secs  = None   # set below; passed into Unicorn pipeline

        if ecog_mat_path is not None:
            log.info("=== ECoG Pipeline ===")
            X_ecog, y_ecog = self.ecog_proc.run(ecog_mat_path)

            ecog_flash_secs = self.ecog_proc.ecog_flash_seconds
            log.info(
                "  ECoG flash times exposed: %d events  (%.2f – %.2f s)",
                len(ecog_flash_secs), ecog_flash_secs[0], ecog_flash_secs[-1],
            )

            results_ecog = self.classifier.fit_evaluate(X_ecog, y_ecog, label_map)
            BCIClassifier.print_results(results_ecog, "ECoG (intracranial)")
            all_results["ECoG"] = results_ecog
            del X_ecog, y_ecog
            gc.collect()

        # Unicorn multi-file pipeline
        if unicorn_files is not None:
            if ecog_flash_secs is None:
                raise RuntimeError(
                    "Unicorn multi-file pipeline requires ECoG flash times. "
                    "Run the ECoG pipeline first (pass ecog_mat_path) or "
                    "provide ecog_flash_seconds directly."
                )

            log.info("=== Unicorn EEG Pipeline (%d files) ===", len(unicorn_files))

            # The same dataset of 126 numbers we've used before
            real_labels = np.array([
                1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1,
                3, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1,
                2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1,
                1, 1, 2, 2, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2,
                2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2,
                2, 2, 3, 3, 3, 3,
            ], dtype=np.int32)

            X_uni, y_uni = self.unicorn_proc.run_multi_file(
                unicorn_files      = unicorn_files,
                ecog_flash_seconds = ecog_flash_secs,
                real_labels        = real_labels,
                label_map          = label_map or {1: "color", 2: "shape", 3: "face"},
            )
            log.info(
                "  Concatenated dataset — X: %s  y: %s  class distribution: %s",
                X_uni.shape, y_uni.shape,
                {int(c): int((y_uni == c).sum()) for c in np.unique(y_uni)},
            )

            results_uni = self.classifier.fit_evaluate(X_uni, y_uni, label_map)
            BCIClassifier.print_results(results_uni, "Unicorn EEG (scalp, 6-file)")
            all_results["Unicorn"] = results_uni
            del X_uni, y_uni
            gc.collect()

        if len(all_results) == 2:
            self._compare(all_results)

        return all_results

    @staticmethod
    def _compare(results: dict) -> None:
        """Print a side-by-side modality comparison table."""
        print("\n" + "=" * 60)
        print("  MODALITY COMPARISON (RandomForest, CV-Accuracy)")
        print("=" * 60)
        for modality, res in results.items():
            rf = res["RandomForest"]
            svm = res["SVM_RBF"]
            print(
                f"  {modality:<12}  RF: {rf['accuracy_mean']:.4f}±{rf['accuracy_std']:.4f}"
                f"   SVM: {svm['accuracy_mean']:.4f}±{svm['accuracy_std']:.4f}"
            )



if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="BCI Dual-Pipeline: ECoG (Walk.mat) vs Unicorn EEG"
    )
    parser.add_argument(
        "--ecog",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to Walk.mat  (required for ECoG pipeline)",
    )
    parser.add_argument(
        "--unicorn",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to Unicorn CSV  (optional; skip Unicorn pipeline if absent)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        metavar="N",
        help="Epochs per RAM chunk during feature extraction (default: 16). "
             "Lower this if you hit OOM on 8 GB with 160-channel ECoG.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of StratifiedKFold folds (default: 5)",
    )
    args = parser.parse_args()


    ECOG_PATH    = args.ecog    or Path("ecog-video/Walk.mat")
    UNICORN_PATH = args.unicorn or None 

    log.info("BCI Dual-Pipeline — Clinical ECoG Run (Walk.mat)")
    log.info("ECoG path    : %s", ECOG_PATH)
    log.info("Unicorn path : %s", UNICORN_PATH or "<not provided — skipping Unicorn pipeline>")

    if not ECOG_PATH.exists():
        raise FileNotFoundError(
            f"Walk.mat not found at: {ECOG_PATH.resolve()}\n"
            "Pass the correct path with: python bci_pipeline.py --ecog /path/to/Walk.mat"
        )

    epoch_cfg = EpochConfig(
        tmin=-0.2,
        tmax=1.0,
        baseline=(-0.2, 0.0),
        chunk_size=args.chunk_size,
    )

    ecog_cfg = ECoGConfig(
        sfreq_expected    = 1200.0,
        mat_variable      = "y",
        n_ecog_channels   = 160,
        ecog_col_start    = 1,
        ecog_col_stop     = 161,
        photodiode_col    = 161,
        stimcode_col      = 162,
        stimcode_video    = 2,
        photodiode_thresh = 0.5,
        notch_freqs       = [50.0, 100.0, 150.0],
        notch_q           = 30.0,
        hg_band           = (70.0, 150.0),
        fir_n_taps        = 257,
    )

    uni_cfg = UnicornConfig(
        sfreq        = 250.0,
        ab_band      = (8.0, 30.0),
        butter_order = 4,
    )

    label_map = {1: "color", 2: "shape", 3: "face"}

    UNICORN_DIR   = Path("unicorn")
    unicorn_files = [
        (UNICORN_DIR / "1/1RAW.csv",  0.0),
        (UNICORN_DIR / "2/2RAW.csv",  0.0),
        (UNICORN_DIR / "3/3RAW.csv", 15.0),
        (UNICORN_DIR / "4/4RAW.csv", 15.0),
        (UNICORN_DIR / "5/5RAW.csv", 15.0),
        (UNICORN_DIR / "6/6RAW.csv", 15.0),
    ]

    _expected_unicorn_files = unicorn_files
    unicorn_files = [(p, o) for p, o in _expected_unicorn_files if Path(p).exists()]

    _missing = [(p, o) for p, o in _expected_unicorn_files if not Path(p).exists()]
    if _missing:
        log.warning(
            "%d of %d expected Unicorn file(s) not found — pipeline will run on "
            "an INCOMPLETE dataset.  Missing paths:\n%s",
            len(_missing),
            len(_expected_unicorn_files),
            "\n".join(f"    {p}" for p, _ in _missing),
        )

    if not unicorn_files:
        log.warning("No Unicorn CSV files found in %s — skipping Unicorn pipeline.", UNICORN_DIR)
        unicorn_files = None

    experiment = BCIExperiment(
        epoch_cfg   = epoch_cfg,
        ecog_cfg    = ecog_cfg,
        unicorn_cfg = uni_cfg,
        n_cv_splits = args.cv_splits,
    )

    results = experiment.run(
        ecog_mat_path = ECOG_PATH,
        unicorn_files = unicorn_files,
        label_map     = label_map,
    )

    log.info("Pipeline complete.")