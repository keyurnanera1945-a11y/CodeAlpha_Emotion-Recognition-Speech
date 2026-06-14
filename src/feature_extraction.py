import librosa
import numpy as np

def extract_features(file_path):

    try:

        audio, sample_rate = librosa.load(
            file_path,
            sr=22050,
            duration=3,
            offset=0.5
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc = np.mean(mfcc.T, axis=0)

        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate
        )

        chroma = np.mean(chroma.T, axis=0)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate
        )

        mel = np.mean(mel.T, axis=0)

        features = np.hstack([
            mfcc,
            chroma,
            mel
        ])

        return features

    except Exception as e:
        print("Error:", e)
        return None