"""
All project settings live here so every script uses the same values.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ravdess"          # unzipped RAVDESS wav files
FEATURES_DIR = DATA_DIR / "features"            # cached .npy spectrograms
METADATA_CSV = DATA_DIR / "metadata.csv"        # one row per clip

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
# RAVDESS emotion codes (3rd field of the file name).
RAVDESS_EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# "calm" and "neutral" sound almost the same and "calm" does not exist in
# other datasets (CREMA-D, TESS), so we merge calm into neutral -> 7 classes.
MERGE_CALM_INTO_NEUTRAL = True

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
NUM_CLASSES = len(EMOTIONS)

# ----------------------------------------------------------------------------
# Speaker-independent split
# ----------------------------------------------------------------------------
# RAVDESS has 24 actors (odd id = male, even id = female).
# The same actor must NEVER appear in two splits, otherwise the model can
# "recognise the voice" instead of the emotion and accuracy gets inflated.
# Every split has the same number of male and female actors.
TRAIN_ACTORS = list(range(1, 17))    # actors 1..16  -> 16 actors
VAL_ACTORS = [17, 18, 19, 20]        # 4 actors
TEST_ACTORS = [21, 22, 23, 24]       # 4 actors

# ----------------------------------------------------------------------------
# Audio / feature settings
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16000                  # resample everything to 16 kHz
CLIP_SECONDS = 3.0                   # every clip becomes exactly 3 seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
TRIM_TOP_DB = 30                     # silence threshold for trimming

N_FFT = 1024                         # 64 ms window
HOP_LENGTH = 512                     # 32 ms hop
N_MELS = 64                          # mel frequency bins
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH   # 94 time frames
# 64 x 94 instead of 128 x 188: ~4x faster training on a laptop CPU
# (2.8 s -> 0.7 s per batch) and 64 mel bins is standard for speech emotion.

RANDOM_SEED = 42
