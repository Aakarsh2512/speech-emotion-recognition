"""
Step 3 helpers: data augmentation.

RAVDESS is small (~960 training clips), so the CNN overfits quickly.
We use two kinds of augmentation:

1. Waveform augmentation (done OFFLINE once in step 3, cached to disk)
       noise injection, pitch shift, time stretch
2. SpecAugment (done ONLINE on every training batch in step 4)
       random frequency masks and time masks on the spectrogram
"""
import librosa
import numpy as np

from src.features import fix_length

# Pitch shift and time stretch use a phase vocoder. librosa's default window
# (2048 samples = 128 ms at 16 kHz) is too long for speech and smears the
# harmonics; a 32 ms window keeps the pitch contour sharp.
VOCODER_N_FFT = 512
VOCODER_HOP = 128


def add_noise(audio, rng, min_snr_db=15.0, max_snr_db=30.0):
    """Add white noise at a random signal-to-noise ratio (lower SNR = more noise)."""
    snr_db = rng.uniform(min_snr_db, max_snr_db)
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape)
    return (audio + noise).astype(np.float32)


def pitch_shift(audio, sample_rate, rng, max_semitones=2.0):
    """Shift the pitch up or down by up to 2 semitones (never by ~0)."""
    semitones = rng.uniform(0.5, max_semitones)
    if rng.random() < 0.5:
        semitones = -semitones
    shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones,
                                          n_fft=VOCODER_N_FFT, hop_length=VOCODER_HOP)
    return shifted.astype(np.float32)


def time_stretch(audio, rng, min_rate=0.85, max_rate=1.15):
    """Speak faster (rate > 1) or slower (rate < 1), then pad/crop back to 3 s."""
    rate = rng.uniform(min_rate, max_rate)
    stretched = librosa.effects.time_stretch(audio, rate=rate,
                                             n_fft=VOCODER_N_FFT, hop_length=VOCODER_HOP)
    return fix_length(stretched).astype(np.float32)


def spec_augment(spectrogram, rng, num_freq_masks=2, max_freq_width=15,
                 num_time_masks=2, max_time_width=20, fill_value=0.0):
    """
    SpecAugment (Park et al., 2019): hide random horizontal (frequency)
    and vertical (time) stripes so the model cannot rely on one small region.

    spectrogram: 2D array (mel_bins, frames). A masked copy is returned.
    fill_value: 0.0 because we apply it AFTER normalisation (mean = 0).
    """
    masked = spectrogram.copy()
    num_bins = masked.shape[0]
    num_frames = masked.shape[1]

    for _ in range(num_freq_masks):
        width = rng.integers(0, max_freq_width + 1)
        start = rng.integers(0, num_bins - width + 1)
        masked[start:start + width, :] = fill_value

    for _ in range(num_time_masks):
        width = rng.integers(0, max_time_width + 1)
        start = rng.integers(0, num_frames - width + 1)
        masked[:, start:start + width] = fill_value

    return masked
