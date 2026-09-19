"""
STEP 2 - Preprocessing and feature caching

    python scripts/step2_extract_features.py

For every clip: resample to 16 kHz -> trim silence -> pad/crop to 3 s
-> log-mel spectrogram (64 x 94).

Saves to data/features/:
    {split}_waves.npy    fixed-length waveforms  (N, 48000)   -> used for augmentation in step 3
    {split}_mels.npy     log-mel spectrograms    (N, 64, 94)
    {split}_labels.npy   emotion index           (N,)
    {split}_actors.npy   actor id                (N,)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import config
from src.dataset import load_metadata
from src.features import fix_length, load_audio, log_mel_spectrogram, trim_silence


def extract_split(metadata, split):
    split_rows = metadata[metadata["split"] == split]

    waves = np.zeros((len(split_rows), config.CLIP_SAMPLES), dtype=np.float32)
    mels = np.zeros((len(split_rows), config.N_MELS, config.N_FRAMES), dtype=np.float32)
    labels = split_rows["label"].to_numpy().astype(np.int64)
    actors = split_rows["actor"].to_numpy().astype(np.int64)

    row_number = 0
    for relative_path in tqdm(split_rows["path"], desc=f"{split:>5}"):
        audio = load_audio(config.PROJECT_ROOT / relative_path)
        audio = trim_silence(audio)
        audio = fix_length(audio)

        waves[row_number] = audio
        mels[row_number] = log_mel_spectrogram(audio)
        row_number += 1

    np.save(config.FEATURES_DIR / f"{split}_waves.npy", waves)
    np.save(config.FEATURES_DIR / f"{split}_mels.npy", mels)
    np.save(config.FEATURES_DIR / f"{split}_labels.npy", labels)
    np.save(config.FEATURES_DIR / f"{split}_actors.npy", actors)
    print(f"{split:>5}: waves {waves.shape}, mels {mels.shape}")


def plot_preprocessing_example(metadata):
    """Show one clip at every stage of the pipeline."""
    row = metadata[metadata["emotion"] == "angry"].iloc[0]
    path = config.PROJECT_ROOT / row["path"]

    raw_audio = load_audio(path)
    trimmed_audio = trim_silence(raw_audio)
    fixed_audio = fix_length(trimmed_audio)
    log_mel = log_mel_spectrogram(fixed_audio)

    figure, axes = plt.subplots(4, 1, figsize=(10, 10))

    librosa.display.waveshow(raw_audio, sr=config.SAMPLE_RATE, ax=axes[0])
    axes[0].set_title(f"1. Raw audio resampled to 16 kHz ({len(raw_audio) / config.SAMPLE_RATE:.2f} s)")

    librosa.display.waveshow(trimmed_audio, sr=config.SAMPLE_RATE, ax=axes[1])
    axes[1].set_title(f"2. Silence trimmed ({len(trimmed_audio) / config.SAMPLE_RATE:.2f} s)")

    librosa.display.waveshow(fixed_audio, sr=config.SAMPLE_RATE, ax=axes[2])
    axes[2].set_title(f"3. Padded / cropped to {config.CLIP_SECONDS:.0f} s")

    image = librosa.display.specshow(log_mel, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                                     x_axis="time", y_axis="mel", ax=axes[3])
    axes[3].set_title(f"4. Log-mel spectrogram {log_mel.shape} -> CNN input")
    figure.colorbar(image, ax=axes[3], format="%+2.0f dB")

    for axis in axes:
        axis.set_xlim(0, max(len(raw_audio), config.CLIP_SAMPLES) / config.SAMPLE_RATE)

    plt.suptitle(f"Preprocessing pipeline ({row['emotion']}, actor {row['actor']})")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "04_preprocessing_pipeline.png", dpi=110)
    plt.close()


def main():
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    for split in ["train", "val", "test"]:
        extract_split(metadata, split)

    plot_preprocessing_example(metadata)
    print(f"\nFeatures saved to {config.FEATURES_DIR}")


if __name__ == "__main__":
    main()
