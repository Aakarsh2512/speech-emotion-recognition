"""
Step 1 helpers: download RAVDESS and turn the file names into a metadata table.

RAVDESS file name example:  03-01-06-01-02-01-12.wav
    field 1  modality         (03 = audio only)
    field 2  vocal channel    (01 = speech)
    field 3  emotion          (01..08, see config.RAVDESS_EMOTION_CODES)
    field 4  intensity        (01 = normal, 02 = strong)
    field 5  statement        (01 = "Kids are talking by the door",
                               02 = "Dogs are sitting by the door")
    field 6  repetition       (01 or 02)
    field 7  actor            (01..24, odd = male, even = female)
"""
import urllib.request
import zipfile

import pandas as pd

from src import config


def download_ravdess():
    """Download and unzip the RAVDESS speech audio (~200 MB) if it is not there yet."""
    if config.RAW_DIR.exists() and any(config.RAW_DIR.glob("Actor_*/*.wav")):
        print(f"RAVDESS already present in {config.RAW_DIR}")
        return

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.RAW_DIR.parent / "ravdess.zip"

    if not zip_path.exists():
        print("Downloading RAVDESS (~200 MB), this can take a few minutes...")
        urllib.request.urlretrieve(config.RAVDESS_URL, zip_path)

    print("Unzipping...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(config.RAW_DIR)


def parse_filename(file_name):
    """Turn '03-01-06-01-02-01-12.wav' into a dict of labels."""
    parts = file_name.replace(".wav", "").split("-")

    emotion_code = parts[2]
    intensity_code = parts[3]
    statement_code = parts[4]
    repetition_code = parts[5]
    actor_id = int(parts[6])

    emotion = config.RAVDESS_EMOTION_CODES[emotion_code]
    original_emotion = emotion
    if config.MERGE_CALM_INTO_NEUTRAL and emotion == "calm":
        emotion = "neutral"

    if actor_id % 2 == 1:
        gender = "male"
    else:
        gender = "female"

    if intensity_code == "01":
        intensity = "normal"
    else:
        intensity = "strong"

    return {
        "emotion": emotion,
        "original_emotion": original_emotion,
        "intensity": intensity,
        "statement": int(statement_code),
        "repetition": int(repetition_code),
        "actor": actor_id,
        "gender": gender,
    }


def actor_to_split(actor_id):
    """Speaker-independent split: each actor belongs to exactly one split."""
    if actor_id in config.TRAIN_ACTORS:
        return "train"
    if actor_id in config.VAL_ACTORS:
        return "val"
    if actor_id in config.TEST_ACTORS:
        return "test"
    raise ValueError(f"Actor {actor_id} is not assigned to any split")


def build_metadata():
    """Scan every wav file and build one table row per clip."""
    rows = []
    wav_paths = sorted(config.RAW_DIR.glob("Actor_*/*.wav"))

    for wav_path in wav_paths:
        row = parse_filename(wav_path.name)
        # store the path relative to the project so the CSV works on any machine
        row["path"] = wav_path.relative_to(config.PROJECT_ROOT).as_posix()
        row["split"] = actor_to_split(row["actor"])
        row["label"] = config.EMOTIONS.index(row["emotion"])
        rows.append(row)

    metadata = pd.DataFrame(rows)
    column_order = ["path", "split", "actor", "gender", "emotion", "label",
                    "original_emotion", "intensity", "statement", "repetition"]
    return metadata[column_order]


def load_metadata():
    """Read the metadata CSV written by scripts/step1_prepare_data.py."""
    return pd.read_csv(config.METADATA_CSV)


def check_no_speaker_leakage(metadata):
    """Fail loudly if any actor shows up in more than one split."""
    train_actors = set(metadata[metadata["split"] == "train"]["actor"])
    val_actors = set(metadata[metadata["split"] == "val"]["actor"])
    test_actors = set(metadata[metadata["split"] == "test"]["actor"])

    assert len(train_actors & val_actors) == 0, "train/val share an actor!"
    assert len(train_actors & test_actors) == 0, "train/test share an actor!"
    assert len(val_actors & test_actors) == 0, "val/test share an actor!"
    print("OK: no actor appears in more than one split (speaker-independent).")
