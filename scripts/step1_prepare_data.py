"""
STEP 1 - Data preparation and EDA

    python scripts/step1_prepare_data.py

1. Downloads RAVDESS (speech, 24 actors, 1440 clips) if needed.
2. Parses every file name into labels and builds data/metadata.csv.
3. Assigns a speaker-independent train / val / test split.
4. Saves EDA plots to reports/figures/.
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
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src import config
from src.dataset import build_metadata, check_no_speaker_leakage, download_ravdess


def plot_class_distribution(metadata):
    """Bar chart: number of clips per emotion in each split."""
    counts = pd.crosstab(metadata["emotion"], metadata["split"])
    counts = counts.reindex(index=config.EMOTIONS, columns=["train", "val", "test"])

    axis = counts.plot(kind="bar", figsize=(9, 4.5), width=0.8)
    axis.set_title("Clips per emotion and split (speaker-independent)")
    axis.set_xlabel("")
    axis.set_ylabel("Number of clips")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "01_class_distribution.png", dpi=120)
    plt.close()
    return counts


def measure_durations(metadata):
    """Duration of every clip before and after trimming leading/trailing silence."""
    raw_durations = []
    trimmed_durations = []

    for relative_path in tqdm(metadata["path"], desc="Measuring durations"):
        full_path = config.PROJECT_ROOT / relative_path
        raw_durations.append(sf.info(full_path).duration)

        audio, _ = librosa.load(full_path, sr=config.SAMPLE_RATE)
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=config.TRIM_TOP_DB)
        trimmed_durations.append(len(trimmed_audio) / config.SAMPLE_RATE)

    return np.array(raw_durations), np.array(trimmed_durations)


def plot_durations(raw_durations, trimmed_durations):
    plt.figure(figsize=(9, 4))
    plt.hist(raw_durations, bins=40, alpha=0.6, label="raw clip")
    plt.hist(trimmed_durations, bins=40, alpha=0.6, label="after trimming silence")
    plt.axvline(config.CLIP_SECONDS, color="red", linestyle="--",
                label=f"fixed length = {config.CLIP_SECONDS:.0f} s")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of clips")
    plt.title("Clip durations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "02_durations.png", dpi=120)
    plt.close()


def plot_examples(metadata):
    """Waveform + log-mel spectrogram of one clip per emotion (same actor, same sentence)."""
    actor = 1
    rows_to_plot = []
    for emotion in config.EMOTIONS:
        match = metadata[(metadata["actor"] == actor)
                         & (metadata["original_emotion"] == emotion)
                         & (metadata["statement"] == 1)
                         & (metadata["repetition"] == 1)]
        # neutral has no "strong" intensity, so just take the last match
        rows_to_plot.append(match.iloc[-1])

    figure, axes = plt.subplots(len(rows_to_plot), 2, figsize=(12, 2.2 * len(rows_to_plot)))

    for row_index in range(len(rows_to_plot)):
        row = rows_to_plot[row_index]
        audio, sr = librosa.load(config.PROJECT_ROOT / row["path"], sr=config.SAMPLE_RATE)

        wave_axis = axes[row_index][0]
        librosa.display.waveshow(audio, sr=sr, ax=wave_axis)
        wave_axis.set_title(f"{row['emotion']} - waveform", fontsize=9)
        wave_axis.set_xlabel("")

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=config.N_FFT,
                                             hop_length=config.HOP_LENGTH, n_mels=config.N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        mel_axis = axes[row_index][1]
        librosa.display.specshow(log_mel, sr=sr, hop_length=config.HOP_LENGTH,
                                 x_axis="time", y_axis="mel", ax=mel_axis)
        mel_axis.set_title(f"{row['emotion']} - log-mel spectrogram", fontsize=9)
        mel_axis.set_xlabel("")

    plt.suptitle(f"Actor {actor}, sentence 1: 'Kids are talking by the door'")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "03_examples_per_emotion.png", dpi=110)
    plt.close()


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. download
    download_ravdess()

    # 2 + 3. metadata with speaker-independent split
    metadata = build_metadata()
    metadata.to_csv(config.METADATA_CSV, index=False)
    print(f"\nSaved {len(metadata)} rows to {config.METADATA_CSV}")
    check_no_speaker_leakage(metadata)

    # 4. EDA
    print("\nClips per split:")
    print(metadata["split"].value_counts().reindex(["train", "val", "test"]).to_string())

    counts = plot_class_distribution(metadata)
    print("\nClips per emotion and split:")
    print(counts.to_string())

    print("\nGender per split:")
    print(pd.crosstab(metadata["split"], metadata["gender"]).to_string())

    raw_durations, trimmed_durations = measure_durations(metadata)
    plot_durations(raw_durations, trimmed_durations)
    print(f"\nRaw duration:     mean {raw_durations.mean():.2f}s, "
          f"min {raw_durations.min():.2f}s, max {raw_durations.max():.2f}s")
    print(f"Trimmed duration: mean {trimmed_durations.mean():.2f}s, "
          f"min {trimmed_durations.min():.2f}s, max {trimmed_durations.max():.2f}s")
    longer = (trimmed_durations > config.CLIP_SECONDS).mean() * 100
    print(f"{longer:.1f}% of trimmed clips are longer than {config.CLIP_SECONDS}s (will be cropped)")

    plot_examples(metadata)
    print(f"\nFigures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
