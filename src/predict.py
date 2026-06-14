import numpy as np
import joblib

from tensorflow.keras.models import load_model

from feature_extraction import extract_features

model = load_model(
    "../models/best_model.keras"
)

encoder = joblib.load(
    "../label_encoder.pkl"
)

scaler = joblib.load(
    "../scaler.pkl"
)

audio_path = input(
    "Enter Audio File Path: "
)

features = extract_features(
    audio_path
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

print(
    "\nPredicted Emotion:",
    emotion[0]
)