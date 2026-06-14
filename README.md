# Emotion Recognition from Speech

## Project Overview

This project uses Deep Learning and Speech Signal Processing to recognize human emotions from speech audio files.

The model analyzes speech recordings and predicts emotions such as:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

## Dataset

* RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

## Technologies Used

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Scikit-learn
* Streamlit

## Features

* Audio preprocessing
* MFCC feature extraction
* Deep Neural Network model
* Emotion prediction from speech
* Interactive Streamlit web application

## Project Structure

Emotion-Recognition-Speech/
├── dataset/
├── models/
├── src/
│ ├── feature_extraction.py
│ ├── train.py
│ └── predict.py
├── app.py
├── requirements.txt
└── README.md

## Installation

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Train the model

```bash
cd src
python train.py
```

4. Predict emotion

```bash
python predict.py
```

5. Run web application

```bash
streamlit run app.py
```

## Model Performance

* Dataset: RAVDESS
* Feature Extraction: MFCC
* Model: Deep Neural Network (DNN)
* Accuracy: Approximately 65%

## Future Improvements

* CNN and LSTM models
* Data augmentation
* Real-time emotion recognition
* Higher prediction accuracy

## Author

Keyur Nanera
