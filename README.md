Linear regression (supervised) + K-means clustering (unsupervised)

---

<p align="center">
  <a href="https://colab.research.google.com/github/SnehaTanwar006/AQI_prediction/blob/main/AQI_prediction.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"/>
  <img src="https://img.shields.io/badge/ML-Supervised%20%2B%20Unsupervised-blueviolet" alt="ML"/>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Methodology](#-methodology)
  - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Supervised Learning — Linear Regression](#3-supervised-learning--linear-regression)
  - [4. Geospatial Visualization](#4-geospatial-visualization)
  - [5. Unsupervised Learning — K-Means Clustering](#5-unsupervised-learning--k-means-clustering)
- [Results & Evaluation](#-results--evaluation)
- [Environment Setup](#-environment-setup)
- [How to Run](#-how-to-run)
- [AQI Scale Reference](#-aqi-scale-reference)
- [License](#-license)

---

## 🌍 Overview

Air Quality Index (AQI) is a standardised metric governments use to communicate how polluted the air currently is or how polluted it is forecast to become. This project applies both **supervised** and **unsupervised** machine-learning techniques to a global AQI dataset to:

1. **Predict** the overall AQI value from individual pollutant sub-indices using **Linear Regression**.
2. **Group** countries/locations into meaningful air-quality clusters using **K-Means Clustering**.
3. **Visualise** the results on an interactive world map built with **Folium** — including a colour-coded heatmap and cluster markers.

The entire workflow is implemented as a single, self-contained **Jupyter Notebook** that can be run locally or directly in **Google Colab** with one click.

---

## 📂 Dataset

| Field | Description |
|---|---|
| **File** | `AQI-and-Lat-Long-of-Countries.csv` |
| **Source** | Public AQI dataset with per-country latitude / longitude coordinates |
| **Key Columns** | `aqi value`, `co aqi value`, `ozone aqi value`, `no2 aqi value`, `pm2.5 aqi value`, `lat`, `lng` |

> **Note:** The CSV is loaded inside the notebook via its full path (`/content/AQI-and-Lat-Long-of-Countries.csv`). When running on **Google Colab** you should upload the file first (or mount Google Drive). When running locally, update the path to match your system — see [How to Run](#-how-to-run).

---

## 🗂 Project Structure

```
AQI_prediction/
├── AQI_prediction.ipynb   # Main Jupyter Notebook (entire pipeline)
├── AQI_pred.pdf.pdf       # Exported PDF version of the notebook
├── LICENSE                # MIT License
└── README.md              # Project documentation (this file)
```

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and manipulation |
| `scikit-learn` | Linear Regression, K-Means, train/test split, evaluation metrics |
| `seaborn` | Pairplot and correlation heatmap |
| `matplotlib` | Scatter plots and line charts |
| `folium` | Interactive world-map visualisation |
| `folium.plugins.HeatMap` | Continuous AQI intensity heatmap layer |
| `branca` | Custom HTML/CSS legend overlay on the Folium map |

---

## 🧠 Methodology

### 1. Data Loading & Preprocessing

```python
data = pd.read_csv('/content/AQI-and-Lat-Long-of-Countries.csv')
data = data.dropna()                                          # Remove missing values
data.columns = [col.strip().lower() for col in data.columns]  # Normalise column names
```

- Rows with **any** missing value are dropped to ensure clean inputs.
- Column names are stripped of whitespace and lower-cased for consistency.

---

### 2. Exploratory Data Analysis (EDA)

| Visualisation | What it reveals |
|---|---|
| **Pairplot** (`seaborn.pairplot`) | Pairwise relationships and distributions across all numerical features |
| **Correlation Heatmap** (`seaborn.heatmap`) | Pearson correlation coefficients between all features; helps identify multicollinearity and the strongest predictors of AQI |

---

### 3. Supervised Learning — Linear Regression

**Features (X):**

| Pollutant | Column |
|---|---|
| Carbon Monoxide | `co aqi value` |
| Ground-level Ozone | `ozone aqi value` |
| Nitrogen Dioxide | `no2 aqi value` |
| Fine Particulate Matter | `pm2.5 aqi value` |

**Target (y):** `aqi value`

**Train / Test Split:** 80 % training — 20 % test (`random_state=42`)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²: ", r2_score(y_test, y_pred))
```

**Output plots:**
- Scatter plot — *Actual vs. Predicted AQI Values*
- Line chart — *Actual vs. Predicted AQI* (index-aligned test set)

---

### 4. Geospatial Visualisation

An interactive **Folium** world map is built in three layers:

1. **HeatMap layer** — continuous colour intensity derived from each location's `aqi value`.
2. **Custom legend** — HTML/CSS overlay explaining the AQI colour scale (Good → Hazardous).
3. **Cluster markers** — colour-coded `CircleMarker` objects added after K-Means (Step 5).

```python
import folium
from folium.plugins import HeatMap

m = folium.Map(location=[mean_lat, mean_lng], zoom_start=3)
heat_data = [[row['lat'], row['lng'], row['aqi value']] for _, row in data.iterrows()]
HeatMap(heat_data).add_to(m)
folium.LayerControl().add_to(m)
```

---

### 5. Unsupervised Learning — K-Means Clustering

```python
from sklearn.cluster import KMeans

X_clust = data[['aqi value', 'co aqi value', 'ozone aqi value', 'no2 aqi value', 'pm2.5 aqi value']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_clust)
X_clust['cluster_label'] = kmeans.labels_
```

| Cluster Colour | Interpretation |
|---|---|
| 🟢 Green | Low / Clean air quality zone |
| 🟠 Orange | Moderate air quality zone |
| 🔴 Red | High / Hazardous air quality zone |

Each location is plotted on the Folium map as a `CircleMarker` coloured according to its cluster, with a tooltip showing the exact AQI value and cluster ID.

---

## 📊 Results & Evaluation

| Metric | Description |
|---|---|
| **MAE** (Mean Absolute Error) | Average absolute deviation of predicted AQI from actual AQI |
| **R² Score** | Proportion of variance in AQI explained by the four pollutant features (closer to 1.0 is better) |

> Exact numerical results are printed inside the notebook upon execution.

---

## ⚙️ Environment Setup

### Option A — Google Colab *(recommended, zero setup)*

1. Click the **Open in Colab** badge at the top of this README.
2. Upload `AQI-and-Lat-Long-of-Countries.csv` via the **Files** panel on the left.
3. Run all cells (`Runtime → Run all`).

---

### Option B — Local Jupyter Environment

#### Prerequisites
- Python **3.8 +**
- `pip` (comes with Python)

#### Step-by-step

**Step 1 — Clone the repository**

```bash
git clone https://github.com/SnehaTanwar006/AQI_prediction.git
cd AQI_prediction
```

**Step 2 — Create & activate a virtual environment** *(optional but recommended)*

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Step 3 — Install dependencies**

```bash
pip install pandas scikit-learn seaborn matplotlib folium branca
```

> Or install Jupyter as well if you haven't already:
> ```bash
> pip install notebook pandas scikit-learn seaborn matplotlib folium branca
> ```

**Step 4 — Place the dataset**

Download / obtain `AQI-and-Lat-Long-of-Countries.csv` and place it in the project root (or any path you prefer).

**Step 5 — Update the dataset path in the notebook**

Open `AQI_prediction.ipynb` and in **Cell 2**, change:

```python
# Before (Colab path)
data = pd.read_csv('/content/AQI-and-Lat-Long-of-Countries.csv')

# After (local path — adjust to your actual path)
data = pd.read_csv('AQI-and-Lat-Long-of-Countries.csv')   # if CSV is in the same folder
# OR
data = pd.read_csv('/absolute/path/to/AQI-and-Lat-Long-of-Countries.csv')
```

**Step 6 — Launch Jupyter and run the notebook**

```bash
jupyter notebook AQI_prediction.ipynb
```

Inside Jupyter, go to **Kernel → Restart & Run All** to execute every cell from top to bottom.

---

## 🗺 AQI Scale Reference

| AQI Range | Category | Colour on Map |
|---|---|---|
| 0 – 50 | Good | 🟢 Green |
| 51 – 100 | Moderate | 🟡 Yellow |
| 101 – 150 | Unhealthy for Sensitive Groups | 🟠 Orange |
| 151 – 200 | Unhealthy | 🔴 Red |
| 201 – 300 | Very Unhealthy | 🟣 Purple |
| 301 – 500 | Hazardous | 🟤 Maroon |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by <a href="https://github.com/SnehaTanwar006">Sneha Tanwar</a></p>
