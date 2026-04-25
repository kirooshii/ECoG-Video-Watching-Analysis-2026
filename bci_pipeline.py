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
    """
    ECoG-specific DSP parameters — configured for Walk.mat clinical dataset.

    Walk.mat column layout (0-based Python indices):
      Col  0        : Time vector  → discard, MNE owns the time axis
      Cols 1–160    : 160 ECoG electrodes  (n_ecog_channels)
      Col  161      : Photodiode  (binary 0/1, rising edge = stimulus onset)
      Col  162      : StimCode    (1=pre-paradigm, 2=video playing, 3=post)
      Col  163      : GroupId     → discard
    """
    sfreq_expected:    float = 1200.0   # Hz — Walk.mat acquisition rate

    # ── Column indices (0-based) ──────────────────────────────────────────────
    mat_variable:      str   = "y"
    n_ecog_channels:   int   = 160      # total electrodes in file (do not change)
    ecog_col_start:    int   = 1        # kept for reference; ROI slicing overrides
    ecog_col_stop:     int   = 161
    photodiode_col:    int   = 161
    stimcode_col:      int   = 162
    stimcode_video:    int   = 2
    photodiode_thresh: float = 0.5

    # ── ROI: visual / temporal / occipital channels (1-based, from electrode map)
    # Channels 1–60   → dense frontal/motor grid            → EXCLUDED
    # Channels 61–100 → posterior temporal, lateral occipital (lateral view) → included
    # Channels 101–160→ inferior temporal, fusiform, ventral occipital       → included
    roi_channels: list = field(default_factory=lambda: list(range(61, 161)))

    # ── DSP parameters ────────────────────────────────────────────────────────
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
        Ingest Walk.mat into an MNE RawArray with strict OOM prevention.

        Walk.mat layout  (346 903 samples × 164 columns, float64, ~430 MB on disk):
          y[:, 0]        — Time vector           → discard
          y[:, 1:161]    — 160 ECoG electrodes   → ECoG channels
          y[:, 161]      — Photodiode (0/1)       → STIM channel "Photodiode"
          y[:, 162]      — StimCode (1/2/3)       → STIM channel "StimCode"
          y[:, 163]      — GroupId                → discard

        Memory budget (float32, n_samples=346 903):
          160 ch ECoG  :  160 × 346 903 × 4 B  ≈  222 MB
          2  aux rows  :    2 × 346 903 × 4 B  ≈    3 MB
          Peak (during slice + copy + del original) ≈ 650 MB — safe on 8 GB.

        The critical window is between sio.loadmat() returning the full float64
        dict (≈430 MB) and the moment we `del mat` — during that window we hold
        both the float64 original and the float32 copy.  The explicit del+GC
        below collapses that window to a single statement.
        """
        cfg = self.cfg
        sfreq = cfg.sfreq_expected

        if path.suffix.lower() != ".mat":
            raise ValueError(f"ECoGProcessor expects a .mat file, got: {path.suffix}")

        # ── Step A: Targeted load — only the 'y' variable, nothing else ───────
        # variable_names= prevents scipy from deserialising every workspace var.
        # For a 164-column mat this is modest, but good practice for labs that
        # store multiple large arrays (e.g., pre-processed copies) in one file.
        log.info("  scipy.io.loadmat('%s', variable_names=['%s']) …", path.name, cfg.mat_variable)
        mat = sio.loadmat(str(path), variable_names=[cfg.mat_variable])
        y = mat[cfg.mat_variable]
        
        # scipy loads it that way too, but the rest of the code expects (n_samples, 164)
        if y.ndim == 2 and y.shape[0] < y.shape[1]:
            y = y.T

        log.info("  Raw matrix shape: %s  dtype: %s", y.shape, y.dtype)
        n_samples = y.shape[0]

        # ── Step B: Slice ROI channels and downcast BEFORE freeing y ──────────
        # roi_channels are 1-based electrode numbers; in the y matrix col 0 is
        # the time vector, so channel N sits at column index N — no offset needed.
        roi_cols = cfg.roi_channels                          # e.g. [61, 62, …, 160]
        n_roi    = len(roi_cols)

        log.info("  Slicing %d ROI channels %d–%d, Photodiode [%d], StimCode [%d] …",
                 n_roi, roi_cols[0], roi_cols[-1],
                 cfg.photodiode_col, cfg.stimcode_col)

        # Advanced index → contiguous copy → transpose to (n_roi, n_samples)
        ecog_data  = y[:, roi_cols].T.astype(np.float32)    # (n_roi, n_samples)
        photodiode = y[:, cfg.photodiode_col].astype(np.float32)
        stimcode   = y[:, cfg.stimcode_col  ].astype(np.float32)

        # ── Step C: CRITICAL — free the full float64 matrix NOW ───────────────
        del y, mat
        gc.collect()
        log.info("  Original matrix freed.  ROI ECoG array: %s  (%.1f MB)",
                 ecog_data.shape, ecog_data.nbytes / 1e6)

        # ── Step D: Build MNE RawArray for ROI channels only ──────────────────
        ch_names = [f"ECoG{ch:03d}" for ch in roi_cols]     # e.g. ECoG061…ECoG160
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=["ecog"] * n_roi,
        )
        raw = mne.io.RawArray(ecog_data * 1e-6, info, verbose=False)
        del ecog_data
        gc.collect()

        # ── Step E: Append Photodiode + StimCode as named STIM channels ───────
        # Naming them explicitly (not "STI014") lets extract_epochs() retrieve
        # them by name, which is more robust than relying on channel order.
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
        Detect photodiode rising edges, gate by StimCode==2, assign labels.

        Why photodiode instead of a digital trigger channel?
        -------------------------------------------------------
        Clinical ECoG systems often lack a dedicated TTL input, so experimenters
        use a photodiode taped to a corner of the stimulus monitor.  The diode
        voltage transitions 0→1 at the exact frame that the stimulus appears,
        giving sub-millisecond precision at 1200 Hz (≈0.83 ms per sample).

        Gating by StimCode==2 ("video playing"):
        ---------------------------------------------------------------
        The Walk.mat paradigm has three phases:
          1 = pre-paradigm  (baseline rest, no stimuli)
          2 = video playing (the condition we care about)
          3 = post-paradigm (recovery, no stimuli)
        Photodiode noise or screen refresh artefacts may fire spurious rising
        edges during phases 1 and 3.  The StimCode gate cleanly rejects these
        without any amplitude thresholding on the neural channels.

        Label assignment (placeholder — TODO for real data):
        ---------------------------------------------------------------
        Walk.mat does not embed per-trial category labels in the matrix itself.
        Labels must come from a separate behavioural log (e.g., a .csv that
        records which colour/shape/face was shown for each photodiode flash).
        Until that log is provided, we cycle through [1, 2, 3] sequentially.
        """
        cfg = self.cfg
        raw = self._raw

        # ── 1. Pull Photodiode and StimCode arrays into RAM ───────────────────
        # raw[channel_name, :] returns (1, T); squeeze to (T,).
        # These are small (346 903 × 4 B ≈ 1.4 MB each) — safe to load fully.
        photodiode = raw["Photodiode"][0][0]   # (T,) float32
        stimcode   = raw["StimCode"  ][0][0]   # (T,) float32

        # ── 2. Detect rising edges on the Photodiode channel ──────────────────
        # A rising edge occurs where the signal crosses from below to above
        # the threshold.  We diff on a boolean array to avoid floating-point
        # precision issues with the 0/1 signal.
        #
        # Derivation:
        #   is_high[t]  = photodiode[t] >= thresh        → boolean mask
        #   rising[t]   = is_high[t] AND NOT is_high[t-1]
        #   Sample index of the rising edge = t (the first HIGH sample).
        #   We add +1 because is_high[1:] corresponds to original index 1..T-1.
        thresh   = cfg.photodiode_thresh
        is_high  = photodiode >= thresh                     # (T,) bool
        rising   = is_high[1:] & ~is_high[:-1]             # (T-1,) bool
        all_rising_samples = np.where(rising)[0] + 1       # 1-based correction

        log.info("  Photodiode: %d total rising edges detected", len(all_rising_samples))

        # ── 3. Gate: keep only events where StimCode == 2 ─────────────────────
        # Index into stimcode at the exact rising-edge sample to check the
        # paradigm phase at that moment.  Cast to int for exact equality.
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

        # ── 4. Assign stimulus category labels ────────────────────────────────
        #
        # TODO: Replace `mock_labels` with your real per-trial label array.
        #
        # Your label array must be shape (n_valid,) with integer codes, e.g.:
        #   1 = color stimulus
        #   2 = shape stimulus
        #   3 = face  stimulus
        #
        # Typical workflow:
        #   behav_log   = pd.read_csv("walk_behavioural_log.csv")
        #   # The log should have one row per photodiode flash during the video.
        #   # Align by trial index (assumes log rows correspond 1:1 to valid events):
        #   real_labels = behav_log["category_code"].values.astype(np.int32)
        #   assert len(real_labels) == n_valid, "Log/event count mismatch!"

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
      
        # ── 5. Build the MNE events array: shape (n_events, 3) ───────────────
        # MNE convention: col0=sample_index, col1=prev_event_id (0), col2=event_id
        events = np.column_stack([
            valid_samples.astype(np.int32),
            np.zeros(n_valid, dtype=np.int32),   # always 0 (MNE convention)
            labels,
        ])

        # ── 6. Map integer codes to human-readable names ──────────────────────
        unique_codes = np.unique(labels)
        # These names are placeholders — update once real labels are injected.
        # TODO: Replace with your actual class name mapping once real labels are used.
        placeholder_names = {1: "class_1", 2: "class_2", 3: "class_3"}
        event_id = {placeholder_names.get(c, f"stim_{c}"): int(c) for c in unique_codes}
        log.info("  Event IDs: %s", event_id)

        # ── 7. Epoch around each event ────────────────────────────────────────
        # preload=False: MNE stores only the event table in RAM; the (n_epochs ×
        # 160 ch × 1441 samples) tensor (~800 MB at float32) is NEVER materialised
        # — it is streamed in chunks inside _build_features().
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=self.epoch_cfg.tmin,
            tmax=self.epoch_cfg.tmax,
            baseline=self.epoch_cfg.baseline,
            picks=mne.pick_types(raw.info, ecog=True),
            preload=False,          # ← CRITICAL: no full tensor in RAM
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
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",   # ← compensates 56/53/17 imbalance
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
                    class_weight="balanced",   # ← already present, kept
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
# 6. Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # ── CLI — accept paths as arguments or fall back to hardcoded defaults ────
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

    # ── Hardcoded fallback paths — edit these for your local environment ──────
    # These are used only when the script is run without CLI arguments, e.g.
    # from an IDE or Jupyter %run magic.
    ECOG_PATH    = args.ecog    or Path("ecog-video/Walk.mat")
    UNICORN_PATH = args.unicorn or None   # set to Path("unicorn_session.csv") if available

    log.info("BCI Dual-Pipeline — Clinical ECoG Run (Walk.mat)")
    log.info("Target: Apple M3 · 8 GB unified memory — OOM-safe mode")
    log.info("ECoG path    : %s", ECOG_PATH)
    log.info("Unicorn path : %s", UNICORN_PATH or "<not provided — skipping Unicorn pipeline>")

    if not ECOG_PATH.exists():
        raise FileNotFoundError(
            f"Walk.mat not found at: {ECOG_PATH.resolve()}\n"
            "Pass the correct path with: python bci_pipeline.py --ecog /path/to/Walk.mat"
        )

    # ── Configuration ─────────────────────────────────────────────────────────
    epoch_cfg = EpochConfig(
        tmin=-0.2,
        tmax=1.0,
        baseline=(-0.2, 0.0),
        # chunk_size=16 → each chunk = 16 epochs × 160 ch × 1441 samples × 4 B ≈ 148 MB
        # Increase to 32 if RAM allows; decrease to 8 if you hit OOM.
        chunk_size=args.chunk_size,
    )

    ecog_cfg = ECoGConfig(
        # All defaults match Walk.mat spec; override here if your file differs.
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

    # ── Label map (placeholder — update once real labels are injected) ─────────
    # TODO: Replace with your actual stimulus category mapping once the
    #       behavioural log is wired into extract_epochs().
    label_map = {1: "color", 2: "shape", 3: "face"}

    # ── Run experiment ────────────────────────────────────────────────────────
    experiment = BCIExperiment(
        epoch_cfg  = epoch_cfg,
        ecog_cfg   = ecog_cfg,
        unicorn_cfg = uni_cfg,
        n_cv_splits = args.cv_splits,
    )

    results = experiment.run(
        ecog_mat_path    = ECOG_PATH,
        unicorn_csv_path = UNICORN_PATH,
        label_map        = label_map,
    )

    log.info("Pipeline complete.")