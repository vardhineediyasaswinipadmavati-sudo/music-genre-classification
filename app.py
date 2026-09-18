import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide"
)

st.title("Music Genre Classification")
st.write("Predict the genre of a music track using audio features.")


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spotify-tracks.csv")

    features = [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms',
        'popularity',
        'key',
        'mode'
    ]

    target = 'track_genre'

    df = df[features + [target]].dropna()

    df[target] = df[target].replace({
        'alt-rock': 'rock',
        'alternative': 'rock',
        'indie': 'rock',
        'indie-rock': 'rock'
    })

    chosen_genres = [
        'rock',
        'acoustic',
        'afrobeat',
        'ambient',
        'pop'
    ]

    df = df[df[target].isin(chosen_genres)]

    return df, features, target


# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model():

    df, features, target = load_data()

    le = LabelEncoder()

    y = le.fit_transform(df[target])
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return rf, scaler, le, accuracy


# -----------------------------
# Load Model
# -----------------------------
with st.spinner("Loading dataset and training model..."):
    df, features, target = load_data()
    model, scaler, le, accuracy = train_model()


# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Tracks", len(df))

with col2:
    st.metric("Genres", len(le.classes_))

with col3:
    st.metric("Model Accuracy", f"{accuracy:.2%}")


st.write("Genres:", ", ".join(le.classes_))


# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Music Features")

col1, col2 = st.columns(2)

with col1:

    danceability = st.slider(
        "Danceability",
        0.0, 1.0, 0.5
    )

    energy = st.slider(
        "Energy",
        0.0, 1.0, 0.5
    )

    loudness = st.number_input(
        "Loudness (dB)",
        value=-5.0
    )

    speechiness = st.slider(
        "Speechiness",
        0.0, 1.0, 0.1
    )

    acousticness = st.slider(
        "Acousticness",
        0.0, 1.0, 0.5
    )

    instrumentalness = st.slider(
        "Instrumentalness",
        0.0, 1.0, 0.0
    )

    liveness = st.slider(
        "Liveness",
        0.0, 1.0, 0.2
    )

with col2:

    valence = st.slider(
        "Valence",
        0.0, 1.0, 0.5
    )

    tempo = st.number_input(
        "Tempo (BPM)",
        min_value=0.0,
        value=120.0
    )

    duration_ms = st.number_input(
        "Duration (milliseconds)",
        min_value=0.0,
        value=200000.0
    )

    popularity = st.slider(
        "Popularity",
        0, 100, 50
    )

    key = st.number_input(
        "Key",
        min_value=0,
        max_value=11,
        value=5
    )

    mode = st.selectbox(
        "Mode",
        [0, 1]
    )


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Genre"):

    input_data = pd.DataFrame([[
        danceability,
        energy,
        loudness,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo,
        duration_ms,
        popularity,
        key,
        mode
    ]], columns=features)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    predicted_genre = le.inverse_transform(prediction)[0]

    st.success(
        f"Predicted Music Genre: {predicted_genre.upper()}"
    )