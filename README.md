# Music Genre Classification Using Machine Learning

A machine learning application that predicts the genre of a music track based on its audio features.

## Project Overview

This project uses the Spotify Tracks Dataset to classify music into five genres:

- Acoustic
- Afrobeat
- Ambient
- Pop
- Rock

The model uses audio features such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, popularity, key, and mode.

## Machine Learning Model

The project uses a Random Forest Classifier.

### Model Configuration

- Number of estimators: 400
- Maximum depth: 25
- Minimum samples split: 3
- Class weight: Balanced
- Random state: 42

## Dataset

The dataset contains Spotify track information and audio features.

The project uses 13 features for classification:

```text
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
duration_ms
popularity
key
mode