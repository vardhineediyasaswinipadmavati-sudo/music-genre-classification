import streamlit as st
import librosa
import numpy as np
import joblib
import os


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Music Genre AI",
    page_icon="🎵",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: white;
}

h1 {
    color: white !important;
}

h2, h3 {
    color: #e2e8f0 !important;
}

p, label {
    color: #cbd5e1 !important;
}


/* Metric cards */

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 15px;
    padding: 15px;
}

div[data-testid="stMetricValue"] {
    color: #a5b4fc;
}


/* File uploader */

div[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.06);
    border: 2px dashed #6366f1;
    border-radius: 15px;
    padding: 15px;
}


/* Predict button */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: 600;
    background: #6366f1;
    color: white;
}

.stButton > button:hover {
    background: #818cf8;
    color: white;
}


/* Prediction result */

.result-box {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.5);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

.result-box h1 {
    color: #a5b4fc !important;
    font-size: 42px;
    margin: 10px 0;
}

.result-box h3 {
    color: #e2e8f0 !important;
}

.result-box p {
    color: #94a3b8 !important;
}


/* How it works */

.info-box {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 15px;
    padding: 20px;
    min-height: 150px;
}

.info-box h3 {
    color: #a5b4fc !important;
}

.info-box p {
    color: #cbd5e1 !important;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = joblib.load("audio_genre_model.pkl")
scaler = joblib.load("audio_scaler.pkl")
label_encoder = joblib.load("audio_label_encoder.pkl")


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        duration=30
    )

    features = []

    # MFCC - 26 features
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    features.extend(
        np.mean(mfcc, axis=1)
    )

    features.extend(
        np.std(mfcc, axis=1)
    )

    # Chroma - 24 features
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features.extend(
        np.mean(chroma, axis=1)
    )

    features.extend(
        np.std(chroma, axis=1)
    )

    # Spectral features
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(centroid)
    )

    features.append(
        np.std(centroid)
    )

    features.append(
        np.mean(bandwidth)
    )

    features.append(
        np.std(bandwidth)
    )

    features.append(
        np.mean(rolloff)
    )

    features.append(
        np.std(rolloff)
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    features.append(
        np.mean(zcr)
    )

    features.append(
        np.std(zcr)
    )

    # RMS Energy
    rms = librosa.feature.rms(
        y=y
    )

    features.append(
        np.mean(rms)
    )

    features.append(
        np.std(rms)
    )

    # Tempo
    tempo = librosa.beat.beat_track(
        y=y,
        sr=sr
    )[0]

    tempo = np.asarray(
        tempo
    ).flatten()[0]

    features.append(
        float(tempo)
    )

    return features


# ==================================================
# MAIN TITLE
# ==================================================

st.title("🎵 Music Genre AI")

st.write(
    "Upload a song and let Machine Learning predict its genre."
)


# ==================================================
# PROJECT INFORMATION
# ==================================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Audio Features",
        "61"
    )

with col2:
    st.metric(
        "Music Genres",
        "10"
    )


# ==================================================
# UPLOAD SECTION
# ==================================================

st.divider()

st.subheader("Upload Your Audio")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3"]
)


# ==================================================
# PREDICTION
# ==================================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button(
        "Predict Music Genre",
        use_container_width=True
    ):

        try:

            # Keep original extension
            extension = os.path.splitext(
                uploaded_file.name
            )[1].lower()

            temp_file = (
                "temp_audio" + extension
            )

            with open(
                temp_file,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            # Extract features
            features = extract_features(
                temp_file
            )

            features = np.array(
                features
            )

            # Check feature count
            if len(features) != 61:

                st.error(
                    f"Feature extraction error. "
                    f"Expected 61 features but got "
                    f"{len(features)}."
                )

            else:

                # Reshape
                features = features.reshape(
                    1,
                    -1
                )

                # Scale
                features_scaled = scaler.transform(
                    features
                )

                # Predict
                prediction = model.predict(
                    features_scaled
                )

                genre = label_encoder.inverse_transform(
                    prediction
                )[0]

                # Format genre name
                if genre == "hiphop":
                    display_genre = "HIP-HOP"
                else:
                    display_genre = genre.upper()

                # Result
                st.divider()

                st.markdown(
                    f"""
                    <div class="result-box">
                        <h3>Predicted Music Genre</h3>
                        <h1>{display_genre}</h1>
                        <p>
                            Prediction generated using
                            Random Forest Machine Learning.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:

            st.error(
                "Unable to process the audio file."
            )

            st.write(
                str(e)
            )


# ==================================================
# HOW IT WORKS
# ==================================================

st.divider()

st.subheader("How It Works")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="info-box">
            <h3>1. Upload</h3>
            <p>
                Upload a WAV or MP3 audio file
                to the application.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="info-box">
            <h3>2. Extract Features</h3>
            <p>
                Librosa extracts 61 audio features
                from the uploaded song.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="info-box">
            <h3>3. Predict</h3>
            <p>
                A Random Forest model analyzes
                the features and predicts the genre.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# SUPPORTED GENRES
# ==================================================

st.divider()

st.subheader("Supported Genres")

st.write(
    "Blues • Classical • Country • Disco • "
    "Hip-Hop • Jazz • Metal • Pop • Reggae • Rock"
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Music Genre Classification using Machine Learning | "
    "Random Forest • Librosa • Streamlit"
)