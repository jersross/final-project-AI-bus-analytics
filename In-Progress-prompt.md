# Vibe Coding Prompt — Spotify Track Clustering, Descriptive Analytics, Genre Classification & Popularity Prediction App

---

## Prompt

You are an expert Python developer building a data analytics application for a recording studio's A&R and production leadership. Build a **single, self-contained Python script (`app.py`)** that runs directly inside a **Google Colab notebook** using the **Gradio framework**, launched with `demo.launch(share=True)` so it produces a public, shareable Gradio link without any external hosting step. The script must run end-to-end without errors in a fresh Colab runtime, whether `spotify_cleaned.csv` is already present in the Colab working directory or needs to be uploaded via the script's own Colab upload prompt.

This app covers the full analytics arc in one build: **descriptive/correlation analysis**, **unsupervised K-Means clustering**, **automated multi-class genre classification**, and **predictive modeling of popularity** — including feature-importance ranking and two myth-busting research questions.

### Context
Our team (Grace Bosma, Jered Ross, Ryan Brock — MGMT 59000, Purdue Mitch Daniels School of Business) is building a data-backed alternative to instinct-driven studio production decisions. A single track can cost $500–$25,000+ to produce, and a full album $10,000–$100,000+, with mixing and production direction currently guided by experience rather than evidence.

Our overall research question is how audio characteristics determine a track's profile, genre, and ultimately popularity. Per our methodology, we answer this in four integrated layers:
1. **Descriptive** — correlation check across core audio features to catch multicollinearity (`energy` vs `loudness`).
2. **Unsupervised** — K-Means clustering of tracks by audio profile alone to uncover mood/energy profiles.
3. **Automated Genre Classification** — a supervised Classifier (`HistGradientBoostingClassifier`) that determines the predicted `track_genre` based strictly on audio metrics and track metadata (`danceability`, `energy`, `valence`, `tempo`, `acousticness`, `speechiness`, `instrumentalness`, `liveness`, `loudness`, `duration_ms`, `explicit`).
4. **Predictive Popularity Modeling** — a Regressor model (`HistGradientBoostingRegressor`) that takes the audio metrics, metadata, AND the automatically predicted/encoded genre signal to forecast `popularity`.

This app lets a non-technical stakeholder input raw track audio metrics and automatically see: (1) what cluster segment it matches, (2) what genre it belongs to based on those metrics, and (3) what popularity score it is expected to achieve.

---

### Dataset Architecture (`spotify_cleaned.csv`)
Dataset size: ~89,740 cleaned rows, 21 columns.

#### 1. Unsupervised Clustering Features (8 Continuous Features)
Standardized via `StandardScaler`:
* `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `speechiness`, `instrumentalness`, `liveness`.

#### 2. Genre Classification Feature Set (Predicting `track_genre`)
Inputs: `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `speechiness`, `instrumentalness`, `liveness`, `loudness`, `duration_ms`, `explicit`. Target: `track_genre`.

#### 3. Popularity Prediction Feature Set (Predicting `popularity`)
Inputs: All 11 track features above + Out-of-Fold Target Encoded `track_genre` signal (or predicted genre encoding during inference). Target: `popularity`.

---

### Required Analytical Engine & Models

1. **Descriptive Engine**:
   * Pearson correlation matrix across numeric attributes.
   * Distribution plots and `energy` vs. `loudness` collinearity analysis.

2. **Unsupervised K-Means Clustering Engine**:
   * Train K-Means on the 8 standardized continuous audio features.
   * Provide Elbow and Silhouette evaluations ($k=2..12$).
   * Default interactive cluster count $k=6$.
   * Compute cluster profiles, average popularity, dominant genres, and genre purity.

3. **Two-Stage Automated Track Simulator Engine**:
   * **Stage 1 (Genre Inference)**: Predicts the track's genre using the fitted Classifier based on the user's selected metrics.
   * **Stage 2 (Popularity Inference)**: Feeds the selected metrics along with the Stage 1 inferred genre into the fitted Regressor to predict the final popularity score.

4. **Predictive Popularity Model & Myth-Busting**:
   * Train `HistGradientBoostingRegressor` on an 80/20 train/test split.
   * Report $R^2$, MAE, and RMSE on test data.
   * Display Permutation Feature Importance bar chart.
   * Myth 1: Danceability vs. Popularity correlation & scatter plot.
   * Myth 2: Explicit vs. Non-Explicit Welch's t-test ($p$-value, mean delta).

---

### Required Gradio Interface (Google Colab)
Gradio `Blocks` app with tabbed navigation:
* **Tab 1: Executive Overview**
* **Tab 2: Descriptive Exploration**
* **Tab 3: Cluster Evaluation**
* **Tab 4: Cluster Profiles & Genre Alignment**
* **Tab 5: Popularity Drivers (Predictive Analytics)**
* **Tab 6: Two-Stage Track Scenario Simulator**: Sliders for track metrics + Explicit checkbox. On submit, automatically predicts:
  1. Nearest Cluster Segment
  2. Determined/Predicted Genre
  3. Final Predicted Popularity Score

---

### Technical & Deployment Specifications
1. Automatic Colab file detection / upload fallback (`google.colab.files.upload()`).
2. Runtime environment check (`google.colab` check for `share=True`).
3. Single script `app.py`, executable in Colab, accompanied by `requirements.txt`.
