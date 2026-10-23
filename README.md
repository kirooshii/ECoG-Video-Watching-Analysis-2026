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
- Bandpass filter 8–30 Hz (4th-order Butterworth, zero-phase)
- Alpha + Beta band power (8–30 Hz) — high-gamma is not accessible through the skull
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
python bci.py
```

By default this reads the ECoG recording from `ecog-video/Walk.mat` and the six Unicorn sessions from `unicorn/N/NRAW.csv`. Options:

- `--ecog PATH`: path to a different `Walk.mat`
- `--cv-splits N`: number of stratified cross-validation folds (default 5)
- `--chunk-size N`: epochs per RAM chunk during feature extraction (default 16; lower it if you run out of memory)

The Unicorn file list and the per-file time offsets are set in the `if __name__ == "__main__"` block at the bottom of `bci.py`.

---

# Results

All numbers are **5-fold stratified cross-validation** (mean ± std across folds). The baseline is the accuracy of always predicting the most common class.

| Modality | Trials | Majority-class baseline | Random Forest | SVM (RBF) |
|---|---|---|---|---|
| ECoG (intracranial, 100 posterior channels) | 126 | 44.4% (56/126 "color") | **66.6% ± 9.6%** | **61.1% ± 4.9%** |
| Unicorn EEG (scalp, 8 channels, 6 sessions) | 730 | 46.0% (336/730 "color") | 43.7% ± 3.9% | 15.3% ± 3.1% |

Classes: color (56 / 336), shape (53 / 316), face (17 / 78) for ECoG / EEG.

**ECoG decodes the stimulus category.** Both classifiers beat the majority-class baseline by about 17–22 percentage points. Macro F1 is 0.67 (RF) and 0.55 (SVM).

**Scalp EEG does not.** The Random Forest reaches 43.7%, slightly below the 46.0% baseline, with macro F1 0.32, about what a classifier guessing at random would get on 3 classes. With 8 dry electrodes and alpha/beta power features, nothing in the signal separates the categories.

The EEG SVM's 15.3% is **below chance**, and it does not mean the SVM learned an inverted signal. Both classifiers use `class_weight="balanced"`, which weights the rare "face" class about 4× more than "color" or "shape". When the features carry no information, the cheapest way for the SVM to lower that weighted loss is to predict "face" for almost everything. Predicting "face" for every epoch would score 78/730 = 10.7%; the SVM lands just above that. Treat this result as "no signal", the same as the Random Forest.

**Takeaway:** with the same stimuli and the same pipeline, invasive high-gamma activity carries enough information to tell color, shape and face stimuli apart, and consumer scalp EEG does not.

### Notes on the numbers

- The script also prints a "Classification Report (full-data fit)" and confusion matrix for each model. These are fit and scored on the **same** data, so they measure training fit, not generalization (the Random Forest scores 100% on both ECoG and EEG there). They are left out of this README; use only the cross-validated numbers above.
- Using all 160 ECoG channels instead of the posterior 100 gives lower CV accuracy (RF 57.1%, SVM 58.7%; see `logs/log1.txt`).
- The 6 EEG sessions are recordings of the same video, so the same stimulus appears in several sessions. The folds are shuffled across sessions, not split by session.
- Full logs for every run are in `logs/`.
