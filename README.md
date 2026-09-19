# Speech Emotion Recognition (SER) with a CNN

A CNN listens to a short voice clip and classifies the speaker's emotion:
**neutral, happy, sad, angry, fearful, disgust, surprised**.

Audio → log-mel spectrogram → 2D CNN → softmax over 7 emotions.

## Project plan

| Step | What | Status |
|---|---|---|
| 1 | Data preparation + EDA, speaker-independent split | ✅ |
| 2 | Preprocessing (trim, pad/crop, log-mel) + feature caching | ⏳ |
| 3 | Augmentation (noise, pitch shift, time stretch, SpecAugment) | ⏳ |
| 4 | CNN model + training (class weights, early stopping, checkpoints) | ⏳ |
| 5 | Evaluation (accuracy, macro F1, per-class metrics, confusion matrix) | ⏳ |
| 6 | Extensions: CNN-LSTM, leakage experiment, Streamlit demo | ⏳ |

## Dataset

[RAVDESS](https://zenodo.org/records/1188976): 1,440 speech clips, 24 professional actors
(12 male, 12 female), 2 sentences, 8 emotions. `calm` is merged into `neutral` → 7 classes.

### Speaker-independent split

| Split | Actors | Clips |
|---|---|---|
| train | 1–16 | 960 |
| val | 17–20 | 240 |
| test | 21–24 | 240 |

No actor appears in two splits. A random split would put the same voice (even the same
sentence) in train and test, so the model could recognise the *speaker* instead of the
*emotion*, and accuracy would be inflated.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run

```bash
python scripts/step1_prepare_data.py      # downloads RAVDESS, builds data/metadata.csv, EDA plots
```

## Step 1 – EDA

![Class distribution](reports/figures/01_class_distribution.png)
![Durations](reports/figures/02_durations.png)
![Examples](reports/figures/03_examples_per_emotion.png)
