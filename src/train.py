import os
import numpy as np
import joblib

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

from feature_extraction import extract_features

DATASET_PATH = "../dataset"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X = []
y = []

total_files = 0

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(
        DATASET_PATH,
        actor
    )

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            total_files += 1

            emotion_code = file.split("-")[2]

            if emotion_code not in emotion_map:
                continue

            emotion = emotion_map[
                emotion_code
            ]

            file_path = os.path.join(
                actor_path,
                file
            )

            features = extract_features(
                file_path
            )

            if features is not None:

                X.append(features)
                y.append(emotion)

print("\nTotal Files:", total_files)

print("\nEmotion Distribution:")
print(Counter(y))

X = np.array(X)

print("Feature Shape:", X.shape)

scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "../scaler.pkl"
)

encoder = LabelEncoder()

y = encoder.fit_transform(y)

joblib.dump(
    encoder,
    "../label_encoder.pkl"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

input_dim = X.shape[1]

model = Sequential()

model.add(
    Dense(
        1024,
        activation="relu",
        input_shape=(input_dim,)
    )
)

model.add(
    BatchNormalization()
)

model.add(
    Dropout(0.4)
)

model.add(
    Dense(
        512,
        activation="relu"
    )
)

model.add(
    BatchNormalization()
)

model.add(
    Dropout(0.4)
)

model.add(
    Dense(
        256,
        activation="relu"
    )
)

model.add(
    BatchNormalization()
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        8,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "../models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=150,
    batch_size=32,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    f"\nFinal Accuracy: {accuracy*100:.2f}%"
)

model.save(
    "../models/emotion_model.keras"
)

print(
    "\nModel Saved Successfully"
)