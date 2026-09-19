"""
Step 2 helpers: audio preprocessing and log-mel spectrogram extraction.

Pipeline for one clip:
    load (resample to 16 kHz, mono)
    -> trim leading / trailing silence
    -> pad or crop to exactly 3 seconds
    -> log-mel spectrogram, shape (128 mel bins, 188 frames)
"""
import librosa
import numpy as np

from src import config


def load_audio(path):
    """Load a file as a mono float32 waveform at 16 kHz."""
    audio, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def trim_silence(audio):
    """Remove silence at the start and end (everything quieter than 30 dB below the peak)."""
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=config.TRIM_TOP_DB)
    return trimmed_audio


def fix_length(audio):
    """
    Make the waveform exactly CLIP_SAMPLES long.
    Longer clips: keep the centre part.
    Shorter clips: pad zeros equally on both sides so the speech stays centred.
    """
    target = config.CLIP_SAMPLES
    length = len(audio)

    if length > target:
        start = (length - target) // 2
        return audio[start:start + target]

    if length < target:
        total_pad = target - length
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant")

    return audio


def preprocess_audio(audio):
    """Trim + fix length. Used for files and for microphone input in the demo."""
    audio = trim_silence(audio)
    audio = fix_length(audio)
    return audio.astype(np.float32)


def preprocess_file(path):
    return preprocess_audio(load_audio(path))


def log_mel_spectrogram(audio):
    """
    Log-mel spectrogram in decibels, shape (N_MELS, N_FRAMES) = (128, 188).
    Values are relative to the loudest point of the clip, so they lie in [-80, 0] dB.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return log_mel.astype(np.float32)


def compute_norm_stats(train_mels):
    """
    Mean and standard deviation of every mel bin over the TRAINING set only
    (using val/test here would leak information). Shapes: (N_MELS, 1).
    """
    mean = train_mels.mean(axis=(0, 2)).reshape(-1, 1)
    std = train_mels.std(axis=(0, 2)).reshape(-1, 1)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(mels, mean, std):
    """Standardise each mel bin to mean 0, std 1. Works for one spectrogram or a batch."""
    return ((mels - mean) / (std + 1e-6)).astype(np.float32)
