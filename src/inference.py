"""
Predict the emotion of a single audio clip (used by the Streamlit demo).
"""
import keras
import numpy as np

from src import config
from src.features import log_mel_spectrogram, normalize, preprocess_audio


def load_predictor(model_name="cnn"):
    model = keras.models.load_model(config.MODELS_DIR / f"{model_name}.keras")
    stats = np.load(config.MODELS_DIR / "norm_stats.npz")
    return model, stats["mean"], stats["std"]


def predict_emotion(audio, model, mean, std):
    """
    audio: mono waveform already at 16 kHz (any length).
    Returns (probabilities for the 7 emotions, the log-mel spectrogram that was used).
    """
    audio = preprocess_audio(audio)
    log_mel = log_mel_spectrogram(audio)
    model_input = normalize(log_mel, mean, std)[np.newaxis, :, :, np.newaxis]   # (1, 64, 94, 1)
    probabilities = model.predict(model_input, verbose=0)[0]
    return probabilities, log_mel
