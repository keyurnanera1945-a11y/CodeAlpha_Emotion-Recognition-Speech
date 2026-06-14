import tempfile
import numpy as np
import streamlit as st
import joblib

from tensorflow.keras.models import load_model

from src.feature_extraction import extract_features

model = load_model(
    "models/emotion_model.keras"
)

encoder = joblib.load(
    "label_encoder.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

st.title(
    "Emotion Recognition from Speech"
)

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        temp_path = temp_file.name

    features = extract_features(
        temp_path
    )

    features = scaler.transform(
        [features]
    )

    prediction = model.predict(
        features,
        verbose=0
    )

    emotion = encoder.inverse_transform(
        [np.argmax(prediction)]
    )

    st.success(
        f"Predicted Emotion: {emotion[0]}"
    )