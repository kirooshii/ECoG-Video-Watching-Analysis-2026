"""
=============================================================================
BCI Dual-Pipeline: ECoG vs. Unicorn Hybrid Black — Comparative Classifier
=============================================================================
Author:  Neuro-Data Engineering Template
Target:  Apple M3 · 8 GB unified memory
Deps:    mne>=1.7, scipy>=1.13, scikit-learn>=1.5, numpy>=1.26, h5py>=3.11

Design philosophy
-----------------
OOM on M3/8 GB is the primary engineering constraint. Every stage therefore
follows three rules:
  1. LAZY LOAD  – raw files are memory-mapped via MNE (preload=False), so
                  only metadata + requested slices hit RAM.
  2. CHUNK      – epoching and feature extraction iterate in fixed-size
                  batches; the full epoch tensor is never materialised.
  3. GC EXPLICIT– del + gc.collect() after every major allocation so the
                  Python GC doesn't wait for the next collection cycle.

DSP conventions
---------------
* All filters are zero-phase (filtfilt / MNE forward–backward) to avoid
  group-delay artefacts that would shift neural response latency.
* Notch filter Q=30 → bandwidth ≈ powerline_freq/30 ≈ 1.67 Hz @50 Hz —
  narrow enough to preserve broadband signal, wide enough to kill harmonics.
* CAR (Common Average Reference) approximates a Laplacian on dense arrays
  and suppresses common-mode noise. Applied AFTER notch to avoid referencing
  noise back in.
* High-Gamma envelope: bandpass FIR (zero-phase, Kaiser window) → |Hilbert|.
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

# ── Silence MNE's verbose channel-type warnings in pipeline contexts ──────────
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bci_pipeline")


# =============================================================================
# 0. Configuration dataclasses — single source of truth for all hyperparameters
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
    """ECoG-specific DSP parameters."""
    sfreq_expected: float = 1000.0   # Hz — typical clinical ECoG
    notch_freqs: list = field(default_factory=lambda: [50.0, 100.0, 150.0])
    notch_q: float = 30.0            # Quality factor → BW = f/Q
    hg_band: Tuple[float, float] = (70.0, 150.0)   # High-Gamma passband
    # FIR filter length for high-gamma: rule of thumb = 3 * (sfreq/low_cutoff)
    # Here: 3 * (1000/70) ≈ 43 → rounded to next odd = 43 taps (≈43 ms group delay,
    # but zero-phase after filtfilt so net delay = 0).
    fir_n_taps: int = 213            # longer → sharper roll-off, higher latency cost
    trigger_channel: str = "TRIGGER" # name of the event/marker channel in .mat


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
    ab_band: Tuple[float, float] = (8.0, 30.0)
    # Butterworth order 4 → −80 dB/decade roll-off; sufficient for Alpha/Beta
    # without excessive filter ringing on the short 1.2 s epoch window.
    butter_order: int = 4
    trigger_column: str = "Trigger"  # column name in the Unicorn CSV export


# =============================================================================
# 1. Abstract base class — defines the contract every processor must fulfil
# =============================================================================

class SignalProcessor(abc.ABC):
    """
    Abstract base for a single-modality BCI preprocessing pipeline.

    Concrete subclasses implement:
      * load_raw()      → mne.io.BaseRaw  (lazy, preload=False)
      * preprocess()    → mne.io.BaseRaw  (filtered, referenced, still lazy)
      * extract_epochs()→ Generator yielding (X_chunk, y_chunk) pairs
    """

    def __init__(self, epoch_cfg: EpochConfig):
        self.epoch_cfg = epoch_cfg
        self._raw: Optional[mne.io.BaseRaw] = None

    # ── public interface ──────────────────────────────────────────────────────

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
            Mean band-power per epoch per channel — the feature matrix.
        y : np.ndarray, shape (n_epochs,)
            Integer-encoded class labels.
        """
        log.info("[%s] Loading: %s", self.__class__.__name__, path.name)
        self._raw = self.load_raw(path)

        log.info("[%s] Preprocessing …", self.__class__.__name__)
        self._raw = self.preprocess()

        log.info("[%s] Extracting epochs …", self.__class__.__name__)
        epochs = self.extract_epochs()

        log.info("[%s] Building feature matrix …", self.__class__.__name__)
        X, y = self._build_features(epochs)

        # Explicit cleanup — frees the memory-mapped file descriptors
        del epochs
        del self._raw
        self._raw = None
        gc.collect()

        log.info("[%s] Done — X: %s  y: %s", self.__class__.__name__, X.shape, y.shape)
        return X, y

    # ── shared helper — chunked feature extraction ────────────────────────────

    def _build_features(
        self, epochs: mne.Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iterate over epochs in fixed-size chunks to avoid materialising the
        full (n_epochs × n_channels × n_times) tensor.

        Feature definition: mean squared amplitude over the epoch window per
        channel = mean instantaneous power (band-limited by preprocessing).
        For ECoG this is mean high-gamma envelope²; for Unicorn it is mean
        alpha/beta power.  Squaring converts amplitude → power units.

        Shape contract: each chunk contributes rows to X of shape (chunk, C);
        stacking yields (n_epochs, n_channels).
        """
        # drop_bad() validates epoch integrity (annotations, rejection thresholds)
        # without loading raw data into RAM. Must precede len() when preload=False,
        # because MNE can't know n_epochs until bad epochs are dropped.
        epochs.drop_bad(verbose=False)

        X_parts, y_parts = [], []
        chunk = self.epoch_cfg.chunk_size
        n_total = len(epochs)

        for start in range(0, n_total, chunk):
            stop = min(start + chunk, n_total)
            # epochs[start:stop].get_data() → (batch, channels, times)
            # Using copy=False avoids an extra allocation where MNE allows it.
            data_chunk = epochs[start:stop].get_data(copy=False)  # (B, C, T)

            # Mean power = mean(signal²) over time axis → (B, C)
            power = np.mean(data_chunk ** 2, axis=-1)
            X_parts.append(power)
            y_parts.append(epochs[start:stop].events[:, 2])  # event id column

            # Release the chunk immediately
            del data_chunk
            gc.collect()

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        return X, y


# =============================================================================
# 2. ECoG Processor
# =============================================================================

class ECoGProcessor(SignalProcessor):
    """
    Preprocessing pipeline for intracranial ECoG recordings stored in MATLAB
    .mat files (v7.3 / HDF5 or legacy v5 via scipy.io).

    Pipeline order matters:
      load → notch → CAR → HG bandpass → Hilbert envelope → epoch

    CAR before bandpass: referencing in broadband prevents the HG bandpass
    from creating reference-channel–specific spectral artefacts.
    """

    def __init__(self, epoch_cfg: EpochConfig, ecog_cfg: ECoGConfig):
        super().__init__(epoch_cfg)
        self.cfg = ecog_cfg

    # ── 2a. Loading ───────────────────────────────────────────────────────────

    def load_raw(self, path: Path) -> mne.io.BaseRaw:
        """
        Ingest a .mat file containing a continuous ECoG recording.

        Expected .mat structure (adjust key names to your lab convention):
          mat['data']    : (n_channels, n_samples) float64
          mat['sfreq']   : scalar sampling frequency
          mat['trigger'] : (1, n_samples) integer trigger channel

        For memory efficiency we use scipy.io with mmap_mode='r' for legacy
        .mat files.  For HDF5-based v7.3 mats, use h5py with lazy slicing.
        """
        suffix = path.suffix.lower()
        if suffix != ".mat":
            raise ValueError(f"ECoGProcessor expects a .mat file, got: {suffix}")

        # scipy.io.loadmat does NOT support mmap_mode for v5 .mat files.
        # True lazy loading is only available via h5py for v7.3 (HDF5) .mat
        # files — the h5py branch below handles that case.
        # For v5 .mat files, we load into float32 immediately and free the
        # original dict to minimise peak RAM (float32 = half footprint of f64).
        try:
            mat = sio.loadmat(str(path))
            data = mat["data"]       # (n_ch, n_times)
            sfreq = float(np.squeeze(mat.get("sfreq", self.cfg.sfreq_expected)))
            trigger_arr = np.squeeze(mat.get(self.cfg.trigger_channel, np.zeros(data.shape[1])))
        except NotImplementedError:
            # v7.3 .mat files are HDF5 — scipy.io cannot handle them
            import h5py  # optional dep; install with: pip install h5py
            with h5py.File(path, "r") as f:
                # HDF5 datasets are lazy by default — only sliced on access
                data = f["data"][:]      # unavoidable full load for MNE compat
                sfreq = float(np.squeeze(f.get("sfreq", self.cfg.sfreq_expected)))
                trigger_arr = np.squeeze(f.get(self.cfg.trigger_channel, np.zeros(data.shape[1])))

        # Build channel names: "ECoG001", "ECoG002", …
        n_ch = data.shape[0]
        ch_names = [f"ECoG{i+1:03d}" for i in range(n_ch)]
        ch_types = ["ecog"] * n_ch

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        # MNE RawArray accepts (n_ch, n_times) — units must be SI (V); ECoG
        # data is often stored in µV, so scale accordingly.
        raw = mne.io.RawArray(data.astype(np.float32) * 1e-6, info, verbose=False)

        # Attach trigger channel as a stimulus channel so MNE can find events
        trig_info = mne.create_info(["STI014"], sfreq, ch_types=["stim"])
        trig_raw = mne.io.RawArray(trigger_arr[np.newaxis, :].astype(np.float32), trig_info, verbose=False)
        raw.add_channels([trig_raw], force_update_info=True)

        # preload=False → data stays on disk / mmap, only headers are in RAM
        # RawArray is always in memory, so we immediately downcast to float32
        # to halve the footprint vs float64.
        del mat, data, trigger_arr
        gc.collect()
        return raw

    # ── 2b. Preprocessing ─────────────────────────────────────────────────────

    def preprocess(self) -> mne.io.BaseRaw:
        """
        Step 1 — Notch filter at powerline frequency and harmonics.
        Step 2 — Common Average Reference (CAR).
        Step 3 — High-Gamma bandpass (FIR, zero-phase) + Hilbert envelope.
        """
        raw = self._raw
        ecog_picks = mne.pick_types(raw.info, ecog=True)

        # ── Step 1: Notch filter ──────────────────────────────────────────────
        # IIR notch via scipy applied in-place channel-by-channel to avoid
        # building a (n_ch, n_times) float64 array in one shot.
        log.info("  Applying notch filters at: %s Hz", self.cfg.notch_freqs)
        for freq in self.cfg.notch_freqs:
            b_notch, a_notch = sig.iirnotch(
                w0=freq,
                Q=self.cfg.notch_q,
                fs=raw.info["sfreq"],
            )
            # Process one channel at a time → peak RAM = 2 × (1 × n_times × 4 B)
            for ch_idx in ecog_picks:
                ch_data, _ = raw[ch_idx, :]          # (1, T)
                filtered = sig.filtfilt(b_notch, a_notch, ch_data[0])
                raw._data[ch_idx, :] = filtered.astype(np.float32)
            gc.collect()

        # ── Step 2: Common Average Reference (CAR) ────────────────────────────
        # CAR subtracts the instantaneous mean across all ECoG electrodes from
        # each electrode. Equivalent to a spatial high-pass filter that removes
        # volume-conducted far-field potentials and reference noise.
        # We compute the mean row-wise to avoid a full copy: CAR = X - mean(X, axis=0)
        log.info("  Applying Common Average Reference (CAR) …")
        # Load only ECoG channels into RAM for this computation
        ecog_data = raw._data[ecog_picks, :]   # view, not copy (float32)
        car_mean = ecog_data.mean(axis=0, keepdims=True)   # (1, T)
        raw._data[ecog_picks, :] -= car_mean   # in-place
        del car_mean
        gc.collect()

        # ── Step 3: High-Gamma bandpass + Hilbert envelope ────────────────────
        # Why FIR over IIR here?
        #   • FIR filters have linear phase → zero-phase after filtfilt, so
        #     temporal alignment of the envelope to triggers is preserved.
        #   • IIR Butterworth at 70–150 Hz on a 1 kHz signal would require very
        #     high order for the steep roll-off needed to exclude Beta bleedthrough.
        # Kaiser window: β=8.6 → −80 dB stop-band attenuation; good for BCI
        # where 60–70 Hz noise can alias into the HG band.
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
            # Hilbert transform: analytic signal → take |·| for amplitude envelope
            # np.abs of complex Hilbert output = instantaneous amplitude
            envelope = np.abs(sig.hilbert(filtered)).astype(np.float32)
            raw._data[ch_idx, :] = envelope
            del filtered, envelope

        gc.collect()
        return raw

    # ── 2c. Epoching ──────────────────────────────────────────────────────────

    def extract_epochs(self) -> mne.Epochs:
        """
        Find events in the STIM channel and segment the continuous signal.

        MNE Epochs with preload=False keeps epoch metadata in RAM but defers
        data loading until iteration — critical for long ECoG recordings.
        """
        events = mne.find_events(
            self._raw,
            stim_channel="STI014",
            verbose=False,
        )
        if events.shape[0] == 0:
            raise ValueError("No events found in STI014 — check trigger channel encoding.")

        # event_id maps descriptive label → integer code stored in events[:,2]
        # Adapt to your actual trigger codes; here we use whatever codes are present.
        unique_codes = np.unique(events[:, 2])
        event_id = {f"stim_{c}": int(c) for c in unique_codes}
        log.info("  Found events: %s", event_id)

        epochs = mne.Epochs(
            self._raw,
            events=events,
            event_id=event_id,
            tmin=self.epoch_cfg.tmin,
            tmax=self.epoch_cfg.tmax,
            baseline=self.epoch_cfg.baseline,
            picks=mne.pick_types(self._raw.info, ecog=True),
            preload=False,    # ← CRITICAL: no full tensor in RAM
            reject_by_annotation=True,
            verbose=False,
        )
        return epochs


# =============================================================================
# 3. Unicorn Hybrid Black Processor
# =============================================================================

class UnicornProcessor(SignalProcessor):
    """
    Preprocessing pipeline for 8-channel consumer EEG (Unicorn Hybrid Black).

    Data may come from:
      (a) Unicorn Suite CSV export  — ``load_from_csv()``
      (b) Lab Streaming Layer (LSL) — ``load_from_lsl()``

    The processor auto-detects based on file extension.

    Pipeline order:
      load → bandpass (Alpha+Beta: 8–30 Hz) → epoch
    No CAR: with only 8 channels CAR would over-suppress genuine signals and
    introduce strong spatial cross-talk between channels.
    """

    def __init__(self, epoch_cfg: EpochConfig, unicorn_cfg: UnicornConfig):
        super().__init__(epoch_cfg)
        self.cfg = unicorn_cfg

    # ── 3a. Loading ───────────────────────────────────────────────────────────

    def load_raw(self, path: Path) -> mne.io.BaseRaw:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv(path)
        elif suffix in (".xdf", ".lsl"):
            return self._load_lsl(path)
        else:
            raise ValueError(f"UnicornProcessor: unsupported format '{suffix}'. Use .csv or .xdf")

    def _load_csv(self, path: Path) -> mne.io.BaseRaw:
        """
        Parse a Unicorn Suite CSV export.
        Expected columns: channel names as header, last column = Trigger.
        Unicorn exports µV-scaled EEG; we convert to Volts for MNE.

        We read with numpy's genfromtxt using a generator to avoid loading the
        entire CSV into RAM at once — important for multi-hour recordings.
        """
        import csv

        log.info("  Reading CSV header …")
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        # Identify EEG columns (all except trigger)
        eeg_cols = [h for h in header if h.strip() not in (self.cfg.trigger_column, "")]
        trig_col_idx = header.index(self.cfg.trigger_column) if self.cfg.trigger_column in header else -1

        # numpy.loadtxt is C-backed and streams from disk → modest RAM usage
        log.info("  Loading CSV data (streaming via numpy) …")
        raw_csv = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            dtype=np.float32,   # float32 halves memory vs float64
        )   # (n_samples, n_cols)

        eeg_indices = [header.index(c) for c in eeg_cols]
        eeg_data = raw_csv[:, eeg_indices].T * 1e-6   # (n_ch, T), µV → V

        # Build MNE info
        ch_names = self.cfg.channel_names[: len(eeg_cols)]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.cfg.sfreq,
            ch_types=["eeg"] * len(ch_names),
        )

        raw = mne.io.RawArray(eeg_data.astype(np.float32), info, verbose=False)
        # Mark as average-referenced after creation (set_eeg_reference needs a Raw instance)
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

        # Add trigger channel if present
        if trig_col_idx != -1:
            trig_data = raw_csv[:, trig_col_idx][np.newaxis, :].astype(np.float32)
            trig_info = mne.create_info(["STI014"], self.cfg.sfreq, ch_types=["stim"])
            trig_raw = mne.io.RawArray(trig_data, trig_info, verbose=False)
            raw.add_channels([trig_raw], force_update_info=True)

        del raw_csv, eeg_data
        gc.collect()
        return raw

    def _load_lsl(self, path: Path) -> mne.io.BaseRaw:
        """
        Load an XDF file recorded via Lab Streaming Layer.
        Requires: pip install pyxdf mne
        """
        try:
            import pyxdf
        except ImportError as exc:
            raise ImportError("Install pyxdf: pip install pyxdf") from exc

        # pyxdf.load_xdf streams per-chunk — the data list is still fully
        # loaded after this call, but XDF files are typically short BCI sessions.
        streams, _ = pyxdf.load_xdf(str(path))

        # Heuristic: find the stream with 8 channels at ~250 Hz
        eeg_stream = next(
            (s for s in streams if int(s["info"]["channel_count"][0]) == 8),
            None,
        )
        if eeg_stream is None:
            raise ValueError("No 8-channel EEG stream found in XDF file.")

        data = np.array(eeg_stream["time_series"]).T.astype(np.float32) * 1e-6
        sfreq = float(eeg_stream["info"]["nominal_srate"][0])

        info = mne.create_info(
            ch_names=self.cfg.channel_names,
            sfreq=sfreq,
            ch_types=["eeg"] * 8,
        )
        return mne.io.RawArray(data, info, verbose=False)

    # ── 3b. Preprocessing ─────────────────────────────────────────────────────

    def preprocess(self) -> mne.io.BaseRaw:
        """
        Butterworth bandpass filter (Alpha + Beta: 8–30 Hz), zero-phase.

        Why Butterworth over FIR here?
          • Unicorn sfreq=250 Hz → 8–30 Hz is a wide relative band; Butterworth
            order 4 gives −80 dB/decade with minimal ringing in the 1.2 s window.
          • FIR at 8 Hz lower-cutoff would require ~3*(250/8) ≈ 94 taps → 376 ms
            of edge effects on a 1200 ms epoch — too costly for the short window.

        Note: we skip CAR. With 8 scalp electrodes, the CAR mean is dominated
        by occipital alpha, which would subtract genuine signal from frontal/
        central channels (C3, Cz, C4) used for motor-imagery decoding.
        """
        sfreq = self._raw.info["sfreq"]
        low, high = self.cfg.ab_band
        nyq = sfreq / 2.0

        log.info("  Bandpass filter: %.1f–%.1f Hz (Butterworth order %d) …",
                 low, high, self.cfg.butter_order)

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

    # ── 3c. Epoching ──────────────────────────────────────────────────────────

    def extract_epochs(self) -> mne.Epochs:
        """
        Identical epoching logic to ECoGProcessor — demonstrates the shared
        contract enforced by the base class.
        """
        try:
            events = mne.find_events(self._raw, stim_channel="STI014", verbose=False)
        except ValueError:
            # Fallback: derive events from annotation if no STIM channel
            events, _ = mne.events_from_annotations(self._raw, verbose=False)

        if events.shape[0] == 0:
            raise ValueError("No events found — check trigger column or annotations.")

        unique_codes = np.unique(events[:, 2])
        event_id = {f"stim_{c}": int(c) for c in unique_codes}

        epochs = mne.Epochs(
            self._raw,
            events=events,
            event_id=event_id,
            tmin=self.epoch_cfg.tmin,
            tmax=self.epoch_cfg.tmax,
            baseline=self.epoch_cfg.baseline,
            picks=mne.pick_types(self._raw.info, eeg=True),
            preload=False,   # lazy — crucial for long Unicorn sessions
            verbose=False,
        )
        return epochs


# =============================================================================
# 4. Machine Learning Pipeline
# =============================================================================

class BCIClassifier:
    """
    sklearn-based classifier wrapper.

    Two estimators are compared:
      • RandomForestClassifier  — robust to feature scale, no tuning needed
      • SVC (RBF kernel)        — strong on small EEG/ECoG feature sets; needs
                                  feature scaling → included in the Pipeline

    Both are wrapped in a ``sklearn.pipeline.Pipeline`` to guarantee that the
    StandardScaler is fit only on training folds (preventing data leakage).

    Cross-validation: StratifiedKFold ensures each fold has the same class
    ratio as the full dataset — essential when class counts are unequal across
    visual stimuli.
    """

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
                    max_depth=None,         # grow fully — pruned by min_samples_leaf
                    min_samples_leaf=2,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )),
            ]),
            "SVM_RBF": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",          # γ = 1/(n_features * X.var())
                    class_weight="balanced",
                    random_state=self.random_state,
                    decision_function_shape="ovr",   # one-vs-rest multiclass
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
        Run StratifiedKFold cross-validation for all classifiers and return
        a structured results dictionary.

        Parameters
        ----------
        X : (n_epochs, n_features)
        y : (n_epochs,) — integer class labels
        label_names : {int: str} mapping for confusion matrix display (optional)

        Returns
        -------
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
            # Final fit on all data for the confusion matrix
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
# 5. Comparative BCI Experiment Orchestrator
# =============================================================================

class BCIExperiment:
    """
    High-level coordinator that runs both modality pipelines and compares
    classifier performance.

    Usage
    -----
    >>> exp = BCIExperiment(epoch_cfg, ecog_cfg, unicorn_cfg)
    >>> exp.run(ecog_mat_path=Path("ecog_recording.mat"),
    ...         unicorn_csv_path=Path("unicorn_session.csv"),
    ...         label_map={1: "color", 2: "shape", 3: "face"})
    """

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
        ecog_mat_path: Optional[Path] = None,
        unicorn_csv_path: Optional[Path] = None,
        label_map: Optional[dict] = None,
    ) -> dict:
        """
        Execute both pipelines and return a dict with all metrics.
        Either pipeline path may be None to run a single-modality experiment.
        """
        all_results = {}

        if ecog_mat_path is not None:
            log.info("=== ECoG Pipeline ===")
            X_ecog, y_ecog = self.ecog_proc.run(ecog_mat_path)
            results_ecog = self.classifier.fit_evaluate(X_ecog, y_ecog, label_map)
            BCIClassifier.print_results(results_ecog, "ECoG (intracranial)")
            all_results["ECoG"] = results_ecog
            del X_ecog, y_ecog
            gc.collect()

        if unicorn_csv_path is not None:
            log.info("=== Unicorn Pipeline ===")
            X_uni, y_uni = self.unicorn_proc.run(unicorn_csv_path)
            results_uni = self.classifier.fit_evaluate(X_uni, y_uni, label_map)
            BCIClassifier.print_results(results_uni, "Unicorn EEG (scalp)")
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


# =============================================================================
# 6. Synthetic data generator — validates the pipeline without real hardware
# =============================================================================

def generate_synthetic_ecog_mat(
    path: Path,
    n_channels: int = 64,
    duration_s: float = 300.0,
    sfreq: float = 1000.0,
    n_classes: int = 3,
    n_trials_per_class: int = 40,
) -> Path:
    """
    Write a synthetic .mat file that mimics a minimal ECoG recording.

    The synthetic signal contains:
      • Pink noise (1/f) as background ECoG baseline
      • Class-specific high-gamma bursts (70–150 Hz) at trial onsets
      • A TRIGGER channel with integer codes 1, 2, 3 at trial onsets

    This lets you validate the full pipeline end-to-end on a laptop with no
    ECoG hardware.
    """
    rng = np.random.default_rng(seed=0)
    n_samples = int(duration_s * sfreq)

    # 1/f noise: FFT-based generation (memory-efficient for large arrays)
    def pink_noise(n: int, n_ch: int) -> np.ndarray:
        f = np.fft.rfftfreq(n)[1:]                   # skip DC
        power = (1.0 / f) ** 0.5
        phases = rng.uniform(0, 2 * np.pi, (n_ch, len(f)))
        spectrum = power * np.exp(1j * phases)
        full_spectrum = np.zeros((n_ch, n // 2 + 1), dtype=complex)
        full_spectrum[:, 1:] = spectrum
        return np.fft.irfft(full_spectrum, n=n).astype(np.float32) * 50  # µV scale

    data = pink_noise(n_samples, n_channels)

    # Inject class-discriminating HG bursts
    trigger = np.zeros(n_samples, dtype=np.float32)
    onset_gap = int(sfreq * (duration_s / (n_classes * n_trials_per_class + 5)))
    onset = int(sfreq * 2.0)   # start 2 s in

    for cls in range(1, n_classes + 1):
        for _ in range(n_trials_per_class):
            if onset + int(sfreq * 1.2) >= n_samples:
                break
            t_burst = np.arange(int(sfreq * 0.5)) / sfreq
            # Different channels respond to different classes (spatial specificity)
            active_chs = slice((cls - 1) * 10, cls * 10)
            burst = (np.sin(2 * np.pi * (80 + cls * 20) * t_burst) * 20).astype(np.float32)
            data[active_chs, onset: onset + len(burst)] += burst[np.newaxis, :]
            trigger[onset] = float(cls)
            onset += onset_gap

    mat_dict = {
        "data": data,           # (n_ch, n_samples) µV
        "sfreq": np.array([[sfreq]]),
        "TRIGGER": trigger[np.newaxis, :],
    }
    sio.savemat(str(path), mat_dict)
    log.info("Synthetic ECoG .mat written: %s  (%d ch × %.0f s)", path, n_channels, duration_s)
    del data, trigger
    gc.collect()
    return path


def generate_synthetic_unicorn_csv(
    path: Path,
    duration_s: float = 300.0,
    sfreq: float = 250.0,
    n_classes: int = 3,
    n_trials_per_class: int = 40,
) -> Path:
    """
    Write a synthetic Unicorn-format CSV with 8 EEG channels + Trigger column.

    Alpha/Beta oscillations (8–30 Hz) are injected class-specifically — 
    mimicking SSVEP or motor-imagery alpha-band modulation.
    """
    rng = np.random.default_rng(seed=1)
    n_samples = int(duration_s * sfreq)
    ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    t = np.arange(n_samples) / sfreq

    # Background: white + some alpha
    data = (rng.standard_normal((8, n_samples)) * 5).astype(np.float32)   # µV
    data += (10 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)          # 10 Hz alpha

    trigger = np.zeros(n_samples, dtype=np.float32)
    onset_gap = int(sfreq * (duration_s / (n_classes * n_trials_per_class + 5)))
    onset = int(sfreq * 2.0)

    for cls in range(1, n_classes + 1):
        for _ in range(n_trials_per_class):
            if onset + int(sfreq * 1.2) >= n_samples:
                break
            t_stim = np.arange(int(sfreq * 1.0)) / sfreq
            freq = 10.0 + cls * 5.0   # class 1→15 Hz, 2→20 Hz, 3→25 Hz
            active = cls - 1          # different channel per class
            data[active, onset: onset + len(t_stim)] += (15 * np.sin(2 * np.pi * freq * t_stim)).astype(np.float32)
            trigger[onset] = float(cls)
            onset += onset_gap

    # Build CSV: header + rows
    header = ch_names + ["Trigger"]
    csv_data = np.vstack([data, trigger[np.newaxis, :]]).T  # (n_samples, 9)
    np.savetxt(
        str(path),
        csv_data,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.6f",
    )
    log.info("Synthetic Unicorn CSV written: %s  (8 ch × %.0f s)", path, duration_s)
    del data, trigger, csv_data
    gc.collect()
    return path


# =============================================================================
# 7. Entry point
# =============================================================================

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    log.info("BCI Dual-Pipeline — Synthetic Validation Run")
    log.info("Target: Apple M3 · 8 GB unified memory — OOM-safe mode")

    # ── Configuration ─────────────────────────────────────────────────────────
    epoch_cfg = EpochConfig(tmin=-0.2, tmax=1.0, baseline=(-0.2, 0.0), chunk_size=32)
    ecog_cfg  = ECoGConfig(sfreq_expected=1200.0, notch_freqs=[50.0, 100.0], hg_band=(70.0, 150.0))
    uni_cfg   = UnicornConfig(sfreq=250.0, ab_band=(8.0, 30.0), butter_order=4)

    label_map = {1: "color", 2: "shape", 3: "face"}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ecog_path   = tmpdir / "ecog_synthetic.mat"
        unicorn_path = tmpdir / "unicorn_synthetic.csv"

        # Generate synthetic data (mimics real recordings structurally)
        generate_synthetic_ecog_mat(
            ecog_path, n_channels=64, duration_s=120.0,
            n_classes=3, n_trials_per_class=30,
        )
        generate_synthetic_unicorn_csv(
            unicorn_path, duration_s=120.0,
            n_classes=3, n_trials_per_class=30,
        )

        # ── Run experiment ────────────────────────────────────────────────────
        experiment = BCIExperiment(
            epoch_cfg=epoch_cfg,
            ecog_cfg=ecog_cfg,
            unicorn_cfg=uni_cfg,
            n_cv_splits=5,
        )
        results = experiment.run(
            ecog_mat_path=ecog_path,
            unicorn_csv_path=unicorn_path,
            label_map=label_map,
        )

    log.info("Pipeline complete. All temp files cleaned up.")
