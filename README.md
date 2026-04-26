# Introduction
This project is done for BR41N.IO Designers' Hackathon during the BCI & Neurotech Spring School 2026, 25-26 April

We are team **G114**, called **Horizons**

We are:
**Makar Lavrov** makar.lavrov.1@iliauni.edu.ge
**Tobías Ezequiel Fleitas Montiel** tobias.montiel.1@iliauni.edu.ge

We are representing **Ilia State University, Tbilisi, Georgia**

in case of any questions be sure to follow up by the email to one of the developers

# BCI Dual-Pipeline

A Python pipeline that compares two types of brain signal recordings to classify what a person is looking at. It processes the signals, extracts features, and evaluates two classifiers side-by-side.

---

## What it does

The experiment shows a participant a series of visual stimulations (126 trials).
Their brain activity is recorded on two devices:

| Device | Type | Channels | Sample rate |
|---|---|---|---|
| ECoG grid | Intracranial (implanted) | 160 electrodes | 1200 Hz |
| Unicorn Hybrid Black | Scalp EEG (non-invasive) | 8 electrodes | 250 Hz |

This pipeline reads both recordings, cleans the signals, and trains classifiers to predict which category of stimulus the person was seeing.

---

## How the pipeline works

```
Raw recording file
       │
       ▼
  Load & memory-map          <- file stays on disk, only small slices hit RAM
       │
       ▼
  Filter the signal          <- remove powerline noise (50 Hz notch), bandpass
       │
       ▼
  Find stimulus triggers     <- photodiode flashes (ECoG) or trigger column (Unicorn)
       │
       ▼
  Cut into epochs            <- –0.2 s to +1.0 s around each flash
       │
       ▼
  Extract features           <- high-gamma power (ECoG) or alpha/beta power (Unicorn)
       │
       ▼
  Train & evaluate           <- Random Forest and SVM, 5-fold cross-validation
```

---

## Signal processing details

**ECoG** (intracranial, high-resolution):
- Notch filter at 50, 100, and 150 Hz to kill powerline interference
- Common Average Reference (CAR) to suppress noise shared across all electrodes
- High-Gamma band (70–150 Hz) envelope via Hilbert transform - this frequency range is strongly linked to local cortical activity
- Only the posterior 100 channels (visual/temporal cortex) are used, the frontal 60 are excluded because of the noise (accuracy is much lower because of the muscle movements)

**Unicorn EEG** (scalp, portable):
- Bandpass filter 1–50 Hz (4th-order Butterworth, zero-phase)
- Alpha (8–13 Hz) and Beta (13–30 Hz) power — high-gamma is not accessible through the skull
- Can combine multiple recording files with a per-file time offset to align with video stimulus timing

---

## Requirements

```
mne >= 1.7
scipy >= 1.13
scikit-learn >= 1.5
numpy >= 1.26
h5py >= 3.11
```

to install:
```
python -m pip install -r requirements.txt
```

---

## Input files

| File | Description |
|---|---|
| `Walk.mat` | ECoG recording. Contains 160 electrode channels, a photodiode channel, and a stimulus code column. |
| `unicorn/N/NRAW.csv` | One CSV per Unicorn recording session. Each file has 8 EEG channels plus a Trigger column. |

---

## Running it

```
python bci_pipeline.py
```

File paths and parameters are configured at the bottom of the script in the `if __name__ == "__main__"` block.

make sure to set:

- `ECOG_MAT_PATH` — path to your `Walk.mat` file
- `UNICORN_DIR` — folder containing the Unicorn CSV subdirectories

---

## Output

The pipeline prints a results table for each device and each classifier:


# Results

## ECoG Results:

```
── RandomForest ──────────────────────────────
  CV Accuracy : 0.6655 ± 0.0956
  Precision   : 0.7044
  Recall      : 0.6586
  F1 (macro)  : 0.6664

  Classification Report (full-data fit):
              precision    recall  f1-score   support

       color       1.00      1.00      1.00        56
       shape       1.00      1.00      1.00        53
        face       1.00      1.00      1.00        17

    accuracy                           1.00       126
   macro avg       1.00      1.00      1.00       126
weighted avg       1.00      1.00      1.00       126

  Confusion Matrix:
[[56  0  0]
 [ 0 53  0]
 [ 0  0 17]]
```

## EEG Results:
```
── RandomForest ──────────────────────────────
  CV Accuracy : 0.4370 ± 0.0386
  Precision   : 0.3114
  Recall      : 0.3289
  F1 (macro)  : 0.3153

  Classification Report (full-data fit):
              precision    recall  f1-score   support

       color       1.00      0.99      1.00       336
       shape       0.99      1.00      1.00       316
        face       1.00      1.00      1.00        78

    accuracy                           1.00       730
   macro avg       1.00      1.00      1.00       730
weighted avg       1.00      1.00      1.00       730

  Confusion Matrix:
[[334   2   0]
 [  1 315   0]
 [  0   0  78]]

── SVM_RBF ──────────────────────────────
  CV Accuracy : 0.1534 ± 0.0305
  Precision   : 0.5619
  Recall      : 0.3610
  F1 (macro)  : 0.1312

  Classification Report (full-data fit):
              precision    recall  f1-score   support

       color       0.65      0.07      0.13       336
       shape       0.83      0.03      0.06       316
        face       0.11      0.97      0.20        78

    accuracy                           0.15       730
   macro avg       0.53      0.36      0.13       730
weighted avg       0.67      0.15      0.11       730

  Confusion Matrix:
[[ 24   2 310]
 [ 11  10 295]
 [  2   0  76]]
 ```