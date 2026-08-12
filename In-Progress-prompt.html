# Vibe Coding Prompt — Spotify Track Clustering, Descriptive Analytics & Popularity Prediction App

---

## Prompt

You are an expert Python developer building a data analytics application for a recording studio's A&R and production leadership. Build a **single, self-contained Python script (`app.py`)** that runs directly inside a **Google Colab notebook** using the **Gradio framework**, launched with `demo.launch(share=True)` so it produces a public, shareable Gradio link without any external hosting step. The script must run end-to-end without errors in a fresh Colab runtime, whether `spotify_cleaned.csv` is already present in the Colab working directory or needs to be uploaded via the script's own Colab upload prompt (see Deployment Requirements below).

This app covers the full analytics arc in one build: **descriptive/correlation analysis**, **unsupervised K-Means clustering**, and **predictive modeling of popularity** — including feature-importance ranking and two myth-busting questions the team's written report calls out specifically. Nothing here is a placeholder for a later phase; all three layers ship together.

### Context
Our team (Grace Bosma, Jered Ross, Ryan Brock — MGMT 59000, Purdue Mitch Daniels School of Business) is building a data-backed alternative to instinct-driven studio production decisions. A single track can cost $500–$25,000+ to produce, and a full album $10,000–$100,000+, with mixing and production direction currently guided by experience rather than evidence.

Our overall research question is which audio characteristics best predict a track's popularity. Per the team's stated methodology, we answer that in three layers, each building on the last:
1. **Descriptive** — a correlation check across core audio features, to catch multicollinearity before it distorts anything built on top of this data.
2. **Unsupervised** — K-Means clustering of tracks by audio profile alone, to test whether Spotify's genre labels are even the right lens for how the studio should think about sound, or whether a mood/energy profile is more useful.
3. **Predictive** — a regression/gradient-boosting model of `popularity` on the audio features and metadata, with feature-importance ranking, plus two myth-busting questions the studio keeps asking: does danceability actually predict popularity, and are explicit tracks more or less popular?

This app is the full deliverable: it lets a non-technical stakeholder explore the descriptive picture, the cluster structure, and what actually drives popularity, all in one place — and see for themselves whether "genre" and "audio-profile cluster" tell the same story, and which audio characteristics the data says matter most.

### Dataset
`spotify_cleaned.csv` — ~89,740 analysis-ready rows (cleaned from an original 114,000-row Kaggle Spotify Tracks Dataset).

21 columns, split below into **clustering features**, **predictive features**, **excluded-but-retained-for-profiling data**, and **identifiers**, with the reasoning for each.

#### Data Kept — Used as Unsupervised Clustering Features (8 Continuous Features)

| Variable | Type | Values / Range | Why Kept |
| --- | --- | --- | --- |
| `danceability` | Continuous | 0.0–1.0 | Core audio-profile signal. |
| `energy` | Continuous | 0.0–1.0 | Core audio-profile signal. |
| `valence` | Continuous | 0.0–1.0 | Core audio-profile signal (musical positivity/mood). |
| `tempo` | Continuous | ~0–250 BPM | Core audio-profile signal; note the wider natural range than the 0–1 features, which is exactly why standardization matters for clustering. |
| `acousticness` | Continuous | 0.0–1.0 | Core audio-profile signal. |
| `speechiness` | Continuous | 0.0–1.0 | Core audio-profile signal. |
| `instrumentalness` | Continuous | 0.0–1.0 | Core audio-profile signal. |
| `liveness` | Continuous | 0.0–1.0 | Core audio-profile signal. |

These 8 continuous features define the standardized "audio profile" a track is clustered on via `StandardScaler`.

#### Data Kept — Expanded Predictive Feature Set (Model Input Target: `popularity`)

To maximize $R^2$ and capture true variance drivers, the predictive model MUST use an expanded feature matrix distinct from the clustering matrix:

| Variable | Type | Values / Range | Why Included in Predictive Model |
| --- | --- | --- | --- |
| `danceability` | Continuous | 0.0–1.0 | Core audio signal. |
| `energy` | Continuous | 0.0–1.0 | Core audio signal. |
| `valence` | Continuous | 0.0–1.0 | Core audio signal. |
| `tempo` | Continuous | ~0–250 BPM | Core audio signal. |
| `acousticness` | Continuous | 0.0–1.0 | Core audio signal. |
| `speechiness` | Continuous | 0.0–1.0 | Core audio signal. |
| `instrumentalness` | Continuous | 0.0–1.0 | Core audio signal. |
| `liveness` | Continuous | 0.0–1.0 | Core audio signal. |
| `loudness` | Continuous | dB, winsorized | Structural production signal. |
| `duration_ms` | Continuous | ms, winsorized | Track duration metadata. |
| `explicit` | Binary / Boolean | 0/1 | Content flag. |
| `track_genre` | Categorical | Target Encoded | Transformed using Out-of-Fold Target Encoding on the training split to capture broad genre/audience reach variance without data leakage. |

*Note: Pitch/scale metadata (`key`, `mode`, `time_signature`) are intentionally excluded from the predictive regressor feature matrix.*

#### Excluded Features (Retained for Descriptive Profiling / Analysis Only)

| Variable | Type | Values / Range | Why Excluded from Clustering & Regressor |
| --- | --- | --- | --- |
| `key` | Nominal (0–11) | Pitch class | Categorical musical metadata; dropped from predictive model to streamline signal, retained for profiling. |
| `mode` | Nominal (0/1) | Major/Minor | Categorical musical metadata; dropped from predictive model, retained for profiling. |
| `time_signature` | Nominal | 3–7 | Categorical musical metadata; dropped from predictive model, retained for profiling. |
| `n_genres_tagged` | Integer | count | Data-cleaning artifact; minor profiling use only. |

#### Identifiers (Display Only — Never Clustering or Predictive Features)
`track_id`, `artists`, `album_name`, `track_name` — used only to label example tracks in the UI.

---

### Required Data Preprocessing
1. Load `spotify_cleaned.csv` with pandas; confirm dtypes (`explicit` is converted to 0/1 integer; `track_genre` is string; continuous audio metrics are float).
2. Set aside identifiers and categorical metadata — **exclude them from the clustering feature matrix**. `popularity` is reserved strictly as the regression target.
3. Compute and display the full Pearson correlation matrix across all numerical attributes (including `loudness`, `duration_ms`, `explicit`, and `popularity`) before model training. The `energy` vs. `loudness` collinearity pair ($r \approx 0.77$) must be explicitly plotted and analyzed in the UI.
4. Standardize all 8 clustering features using `StandardScaler` so `tempo` doesn't dominate Euclidean distance metrics.
5. For the predictive layer, perform an 80/20 train/test split with `random_state=42`. Apply Out-of-Fold Target Encoding to `track_genre` on the training split to prevent target leakage.
6. Comment/justify preprocessing decisions in the code.

---

### Required Analytical Engine
1. **Descriptive Engine**: Compute full Pearson correlation matrix across all numeric features, rendered as a heatmap. Generate individual feature distribution histograms and a dedicated `energy` vs. `loudness` scatter plot.
2. **Unsupervised K-Means Clustering Engine**:
   - Train K-Means on the 8 standardized continuous audio features.
   - Provide Elbow (Inertia) and Silhouette Score evaluations across $k = 2$ through $12$.
   - Set default interactive cluster count to $k = 6$, controllable via UI slider.
   - Compute post-hoc cluster profiles: mean audio metrics, average popularity, dominant genres, and genre purity score (percentage of cluster tracks belonging to its primary genre).
3. **Cluster-vs-Genre Alignment Analysis**:
   - A cluster × top 20 genres crosstab heatmap.
   - A summary alignment metric (e.g., weighted genre purity) quantifying how much cluster assignment and genre labels agree.
4. **Predictive Popularity Model**:
   - **Algorithm**: `scikit-learn`'s `HistGradientBoostingRegressor` (or `GradientBoostingRegressor`).
   - **Split**: 80/20 train/test split with `random_state=42`.
   - **Evaluation**: Display Test $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
   - **Feature Importance**: Extract and display Permutation Feature Importance as a sorted horizontal bar chart.
   - **Myth-Busting Analysis**:
     - *Myth 1 (Danceability vs. Popularity)*: Compute correlation, display scatter plot with trend line, and state an empirical, correlational verdict.
     - *Myth 2 (Explicit Content Impact)*: Perform two-sample Welch's t-test comparing explicit vs. non-explicit track popularity distributions; output p-value, t-statistic, and mean delta.

---

### Required Gradio Interface (Google Colab)
Build a multi-tab Gradio `Blocks` app with tabbed navigation:

1. **Tab 1: Executive Overview**: Business context, methodology, research questions, and workflow guide.
2. **Tab 2: Descriptive Exploration**: Pearson correlation heatmap, feature distributions, and loudness vs. energy scatter plot highlighting collinearity.
3. **Tab 3: Cluster Evaluation**: Elbow curve and Silhouette plots ($k=2..12$), interactive $k$-selection slider, and re-clustering controls.
4. **Tab 4: Cluster Profiles & Genre Alignment**: Detailed cluster metadata summary dataframe, audio profile comparison charts, headline KPIs, and Cluster vs. Genre crosstab heatmap.
5. **Tab 5: Popularity Drivers (Predictive Analytics)**: Test set metrics ($R^2$, RMSE, MAE), permutation feature importance ranking chart, and dedicated plots/verdicts for the two Myth-Busting research questions.
6. **Tab 6: Track Scenario Simulator**: Interactive input sliders for audio features (`danceability`, `energy`, `valence`, `tempo`, `acousticness`, `speechiness`, `instrumentalness`, `liveness`, `loudness`, `duration_ms`), explicit checkbox, and genre dropdown. On submit, computes:
   - Nearest Cluster Assignment (via fitted K-Means centroids).
   - Predicted Popularity Score (via fitted Regressor model).

---

### Technical & Deployment Specifications
1. **Colab File Detection**: Automatically check if `spotify_cleaned.csv` exists in the local directory. If missing, trigger `google.colab.files.upload()`.
2. **Environment Handling**: Detect runtime environment (`'google.colab' in sys.modules`). Trigger `share=True` on `demo.launch()` only inside Colab.
3. **Execution Efficiency**: Pre-calculate scaling, base correlation matrices, and model training at script initialization. Cache state so adjusting UI sliders (like $k$) executes quickly without retraining the predictive regressor.
4. **Deliverables**:
   - Complete, error-free `app.py` executable in a single Colab cell.
   - `requirements.txt` block (`gradio`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`).
   - Setup shell commands for Colab execution.
