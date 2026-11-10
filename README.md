Group: <GROUP_NUMBER>

# 🌦️ Machine Learning Project (364-1-1811) — Part A: Data Collection, Understanding & Preparation

Binary classification pipeline predicting **`RainTomorrow`** (`1` if next day's `precipitation_sum >= 1.0` mm, else `0`) across 6 global cities (`london`, `berlin`, `new_york`, `tokyo`, `sydney`, `vancouver`) using 20 full calendar years (`2006-01-01` to `2025-12-31`) from the **Open-Meteo Historical Weather API** (`ERA5` reanalysis).

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── london.json, berlin.json, new_york.json, tokyo.json, sydney.json, vancouver.json
│   │   └── sensed_data.csv          # Unified deduplicated raw table (43,830 rows, 14 columns)
│   └── processed/
│       └── final_dataset.csv        # Final prepared dataset (43,824 rows, 17 columns)
├── figures/                         # All High-DPI figures for the Notebook & Report
├── data_collection.py               # Task 1.1: Open-Meteo API client with rate-limiting & retry backoff
├── sensing.py                       # Tasks 1.2-1.3 & Section 4: Offline JSON -> CSV sensing & preparation
├── dataset_creation.ipynb           # Section 3 & 4: Full EDA, Sanity Checks, Discretization & Engineering
└── requirements.txt                 # Required Python packages
```

---

## 🚀 How to Run the Pipeline

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Collect raw weather data (`data/raw/*.json`)**:
   ```bash
   python data_collection.py
   ```
   *(Uses local JSON cache if files already exist in `data/raw/`; pass `--force-refresh` to re-download from Open-Meteo).*
3. **Run offline sensing and data preparation (`sensed_data.csv` & `final_dataset.csv`)**:
   ```bash
   python sensing.py
   ```
4. **Explore full analysis, visualizations, and report context**:
   Open `dataset_creation.ipynb` in Jupyter Notebook / VS Code / GitHub.
