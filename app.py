import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Music Genre AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #1e1b4b 100%
        );
        color: white;
    }

    /* Remove default top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(
            90deg,
            #a78bfa,
            #60a5fa,
            #22d3ee
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 40px;
    }

    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 18px;
    }

    /* Genre result */
    .result-card {
        background: linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.25),
            rgba(37, 99, 235, 0.20)
        );
        border: 1px solid rgba(167, 139, 250, 0.5);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
    }

    .result-label {
        color: #cbd5e1;
        font-size: 16px;
        margin-bottom: 8px;
    }

    .result-genre {
        font-size: 42px;
        font-weight: 800;
        color: #a78bfa;
        text-transform: uppercase;
    }

    /* Info cards */
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.10);
    }

    .info-number {
        font-size: 28px;
        font-weight: 700;
        color: #60a5fa;
    }

    .info-text {
        color: #cbd5e1;
        font-size: 14px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.04);
        border: 2px dashed rgba(167, 139, 250, 0.5);
        border-radius: 16px;
        padding: 20px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px 25px;
        font-size: 17px;
        font-weight: 700;
        color: white;
        background: linear-gradient(
            90deg,
            #7c3aed,
            #2563eb
        );
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35);
    }

    /* Audio player */
    audio {
        width: 100%;
        margin-top: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("audio_genre_model.pkl")
    scaler = joblib.load("audio_scaler.pkl")
    encoder = joblib.load("audio_label_encoder.pkl")

    return model, scaler, encoder


model, scaler, encoder = load_model()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        duration=30
    )

    features = []

    # MFCC - 13
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    features.extend(
        np.mean(mfcc, axis=1)
    )

    # Chroma - 12
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features.extend(
        np.mean(chroma, axis=1)
    )

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(spectral_centroid)
    )

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(spectral_bandwidth)
    )

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(spectral_rolloff)
    )

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    features.append(
        np.mean(zero_crossing_rate)
    )

    # RMS Energy
    rms = librosa.feature.rms(y=y)

    features.append(
        np.mean(rms)
    )

    # Tempo
    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    features.append(
        float(np.asarray(tempo).reshape(-1)[0])
    )

    return np.array(features)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Music Genre AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a music track and let machine learning identify its genre'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL INFO
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">31</div>
        <div class="info-text">Audio Features</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">5</div>
        <div class="info-text">Music Genres</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">AI</div>
        <div class="info-text">Random Forest Model</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Upload Your Music</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3"],
    help="Upload a WAV or MP3 music file."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    st.write("")

    if st.button("Predict Music Genre"):

        temp_path = None

        try:

            file_extension = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name

            with st.spinner(
                "Analyzing your music..."
            ):

                features = extract_features(
                    temp_path
                )

                features = features.reshape(
                    1, -1
                )

                features_scaled = scaler.transform(
                    features
                )

                prediction = model.predict(
                    features_scaled
                )

                genre = encoder.inverse_transform(
                    prediction
                )[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Music Genre
                    </div>
                    <div class="result-genre">
                        {genre}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                f"Unable to process the audio file: {e}"
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# =========================================================
# SUPPORTED GENRES
# =========================================================

st.write("")

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Supported Genres</div>',
    unsafe_allow_html=True
)

st.write(
    "Blues  •  Classical  •  Country  •  Jazz  •  Rock"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Music Genre Classification • Machine Learning Project
        <br>
        Built with Python, Librosa, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)