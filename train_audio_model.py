import os
import librosa
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


DATASET_PATH = r"C:\Users\yasaswini\Downloads\archive\Data\genres_original"


# Extract features from one audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))

    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))

    features.append(np.mean(spectral_rolloff))
    features.append(np.std(spectral_rolloff))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # RMS energy
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(np.asarray(tempo).reshape(-1)[0]))

    return features


# Use 5 genres initially
genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

data = []
labels = []

print("Starting feature extraction...\n")

for genre in genres:

    genre_path = os.path.join(DATASET_PATH, genre)

    print(f"Processing {genre}...")

    files = [
        f for f in os.listdir(genre_path)
        if f.endswith(".wav")
    ]

    for file in files:

        file_path = os.path.join(genre_path, file)

        try:
            features = extract_features(file_path)

            data.append(features)
            labels.append(genre)

        except Exception as e:
            print(f"Skipped {file}: {e}")


# Convert to arrays
X = np.array(data)
y = np.array(labels)

print("\nFeature extraction completed.")
print("Samples:", len(X))
print("Features per sample:", X.shape[1])


# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)


# Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("Audio Genre Classification")
print("================================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Genres:", list(encoder.classes_))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save model files
joblib.dump(model, "audio_genre_model.pkl")
joblib.dump(scaler, "audio_scaler.pkl")
joblib.dump(encoder, "audio_label_encoder.pkl")

print("\nModel saved successfully!")