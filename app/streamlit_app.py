"""
STEP 6 - Live demo

    streamlit run app/streamlit_app.py

Record your voice with the microphone or upload a clip, and the model
shows the predicted emotion with the probability for every class.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.inference import load_predictor, predict_emotion

EMOJI = {
    "neutral": "😐",
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲",
}


@st.cache_resource
def get_predictor(model_name):
    return load_predictor(model_name)


def available_models():
    names = []
    for model_file in sorted(config.MODELS_DIR.glob("*.keras")):
        names.append(model_file.stem)
    return names


st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.write("A CNN trained on RAVDESS log-mel spectrograms with a speaker-independent split. "
         "Record a short sentence (2-4 s) or upload a clip.")

model_names = available_models()
if len(model_names) == 0:
    st.error("No trained model found in models/. Run scripts/step4_train.py first.")
    st.stop()

model_name = st.sidebar.selectbox("Model", model_names)
st.sidebar.caption("Trained on 16 actors reading 2 sentences in a studio. "
                   "Real microphones and spontaneous speech are harder, so expect lower accuracy "
                   "than on the test set. Exaggerated, acted emotions work best.")

recorded = st.audio_input("Record with your microphone")
uploaded = st.file_uploader("...or upload an audio file", type=["wav", "mp3", "flac", "ogg"])

audio_bytes = None
if recorded is not None:
    audio_bytes = recorded.getvalue()
elif uploaded is not None:
    audio_bytes = uploaded.getvalue()
    st.audio(audio_bytes)

if audio_bytes is not None:
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=config.SAMPLE_RATE, mono=True)

    if len(audio) < config.SAMPLE_RATE * 0.5:
        st.warning("The clip is shorter than 0.5 s - please record a full sentence.")
        st.stop()

    model, mean, std = get_predictor(model_name)
    probabilities, log_mel = predict_emotion(audio, model, mean, std)

    best_index = int(probabilities.argmax())
    best_emotion = config.EMOTIONS[best_index]
    st.subheader(f"{EMOJI[best_emotion]} {best_emotion.capitalize()}  ({probabilities[best_index]:.0%})")

    chart_data = pd.DataFrame({"probability": probabilities}, index=config.EMOTIONS)
    st.bar_chart(chart_data)

    figure, axis = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(log_mel, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                             x_axis="time", y_axis="mel", ax=axis)
    axis.set_title("What the model sees: log-mel spectrogram (trimmed, 3 s)")
    st.pyplot(figure)
