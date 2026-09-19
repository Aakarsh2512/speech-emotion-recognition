"""
STEP 3 - Offline waveform augmentation (training set only)

    python scripts/step3_augment.py

For every TRAINING clip we create 3 extra versions:
    1. + white noise
    2. pitch shifted (+/- 0.5..2 semitones)
    3. time stretched (0.85x .. 1.15x) + a little noise

So the training set grows from ~960 to ~3840 spectrograms.
Validation and test clips are NEVER augmented - they must stay real.

SpecAugment is not done here; it is applied randomly on every batch
during training (see src/data_generator.py).

Saves to data/features/:
    train_aug_mels.npy     (3 * N_train, 64, 94)
    train_aug_labels.npy   (3 * N_train,)
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
from src.augment import add_noise, pitch_shift, spec_augment, time_stretch
from src.features import log_mel_spectrogram

AUGMENTED_COPIES_PER_CLIP = 3


def make_augmented_versions(audio, rng):
    """Return the 3 augmented waveforms for one clip."""
    noisy = add_noise(audio, rng)
    shifted = pitch_shift(audio, config.SAMPLE_RATE, rng)
    stretched = add_noise(time_stretch(audio, rng), rng, min_snr_db=25.0, max_snr_db=35.0)
    return [noisy, shifted, stretched]


def plot_augmentation_examples(audio, rng):
    original = log_mel_spectrogram(audio)
    noisy = log_mel_spectrogram(add_noise(audio, rng, min_snr_db=15.0, max_snr_db=15.0))
    shifted = log_mel_spectrogram(pitch_shift(audio, config.SAMPLE_RATE, rng))
    stretched = log_mel_spectrogram(time_stretch(audio, rng))
    masked = spec_augment(original, rng, fill_value=original.min())

    images = [original, noisy, shifted, stretched, masked]
    titles = ["Original", "Noise injection (SNR 15 dB)", "Pitch shift",
              "Time stretch", "SpecAugment (freq + time masks)"]

    figure, axes = plt.subplots(len(images), 1, figsize=(9, 12))
    for index in range(len(images)):
        librosa.display.specshow(images[index], sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                                 x_axis="time", y_axis="mel", ax=axes[index])
        axes[index].set_title(titles[index], fontsize=10)
        axes[index].set_xlabel("")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "05_augmentations.png", dpi=110)
    plt.close()


def main():
    rng = np.random.default_rng(config.RANDOM_SEED)

    train_waves = np.load(config.FEATURES_DIR / "train_waves.npy")
    train_labels = np.load(config.FEATURES_DIR / "train_labels.npy")
    num_clips = len(train_waves)

    aug_mels = np.zeros((num_clips * AUGMENTED_COPIES_PER_CLIP, config.N_MELS, config.N_FRAMES),
                        dtype=np.float32)
    aug_labels = np.zeros(num_clips * AUGMENTED_COPIES_PER_CLIP, dtype=np.int64)

    output_index = 0
    for clip_index in tqdm(range(num_clips), desc="Augmenting train clips"):
        versions = make_augmented_versions(train_waves[clip_index], rng)
        for augmented_audio in versions:
            aug_mels[output_index] = log_mel_spectrogram(augmented_audio)
            aug_labels[output_index] = train_labels[clip_index]
            output_index += 1

    np.save(config.FEATURES_DIR / "train_aug_mels.npy", aug_mels)
    np.save(config.FEATURES_DIR / "train_aug_labels.npy", aug_labels)
    print(f"Original training clips: {num_clips}")
    print(f"Augmented spectrograms:  {len(aug_mels)}")
    print(f"Total for training:      {num_clips + len(aug_mels)}")

    plot_augmentation_examples(train_waves[0], rng)
    print(f"Figure saved to {config.FIGURES_DIR / '05_augmentations.png'}")


if __name__ == "__main__":
    main()
