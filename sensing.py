"""
Sensing and Dataset Creation Script for ML Project Part A.

Reads raw JSON files downloaded from Open-Meteo in data/raw/,
unifies them into a tabular structure with added 'location' and 'date' columns,
deduplicates by (location, date), and creates:
1. data/raw/sensed_data.csv: Unified raw tabular data.
2. data/processed/final_dataset.csv: Processed tabular data with target column
   'RainTomorrow' (1 if next day precipitation_sum >= 1.0 mm, else 0),
   excluding the final day of each location to avoid missing targets/leakage.

Note: As per assignment instructions, NO API calls are made from this file.
"""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

LOCATIONS = ["london", "berlin", "new_york", "tokyo", "sydney", "vancouver"]

DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
]


def load_raw_json_files(raw_dir: Path, locations: List[str]) -> pd.DataFrame:
    """
    Reads local JSON files for each location, parses 'daily' dict into a DataFrame,
    adds 'location' and 'date' columns, and combines all locations.
    """
    dataframes: List[pd.DataFrame] = []

    for loc in locations:
        json_path = raw_dir / f"{loc}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing required raw JSON file for location '{loc}' at {json_path}. "
                "Run data_collection.py first."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if "daily" not in payload:
            raise KeyError(f"Key 'daily' missing from JSON payload in {json_path}")

        daily_data = payload["daily"]
        df_loc = pd.DataFrame(daily_data)

        # Standardize date column name from 'time' to 'date'
        if "time" in df_loc.columns:
            df_loc.rename(columns={"time": "date"}, inplace=True)

        df_loc["location"] = loc

        # Ensure location and date are first columns
        ordered_cols = ["location", "date"] + [
            col for col in DAILY_VARIABLES if col in df_loc.columns
        ]
        df_loc = df_loc[ordered_cols]
        dataframes.append(df_loc)
        logging.info("[%s] Loaded %d daily records.", loc, len(df_loc))

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def create_sensed_data(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Deduplicates combined DataFrame by (location, date) and saves sensed_data.csv.
    """
    initial_count = len(df)
    deduped_df = df.drop_duplicates(subset=["location", "date"]).copy()
    duplicates_removed = initial_count - len(deduped_df)

    if duplicates_removed > 0:
        logging.warning("Removed %d duplicate rows by (location, date).", duplicates_removed)
    else:
        logging.info("No duplicate rows found. Total records: %d", len(deduped_df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped_df.to_csv(output_path, index=False)
    logging.info("Saved unified raw dataset to %s (%d rows)", output_path, len(deduped_df))
    return deduped_df


def create_final_dataset(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Sorts by location and date, calculates binary RainTomorrow target
    based on next day's precipitation_sum >= 1.0 mm, drops the final day
    for each location, and saves final_dataset.csv.
    """
    # Sort chronologically within each location
    df_sorted = df.sort_values(by=["location", "date"]).reset_index(drop=True).copy()

    # Calculate tomorrow's precipitation using shift(-1) grouped by location
    next_day_precip = df_sorted.groupby("location")["precipitation_sum"].shift(-1)

    # RainTomorrow: 1 if tomorrow's precipitation >= 1.0 mm, else 0
    df_sorted["RainTomorrow"] = (next_day_precip >= 1.0).astype(int)

    # The last day of each location has NaN for next_day_precip; identify and drop it
    last_day_mask = next_day_precip.isna()
    dropped_count = last_day_mask.sum()
    final_df = df_sorted[~last_day_mask].reset_index(drop=True).copy()

    logging.info(
        "Created 'RainTomorrow' target. Dropped %d last-day records without future labels.",
        dropped_count,
    )

    # --- Section 4 Data Preparation Steps ---
    # 4.2 Discretization (Numeric integer codes preserving order)
    # 1. wind_speed_level (0: <19.5, 1: 19.5-28.5, 2: 28.5-38.5, 3: >=38.5 km/h)
    final_df["wind_speed_level"] = pd.cut(
        final_df["wind_speed_10m_max"],
        bins=[-np.inf, 19.5, 28.5, 38.5, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # 2. wind_dir_sector (8 compass sectors 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW)
    final_df["wind_dir_sector"] = (
        ((final_df["wind_direction_10m_dominant"] + 22.5) % 360) // 45
    ).astype(int)

    # 4.3 Feature Engineering (Derived features from existing columns)
    # 1. precip_intensity = precipitation_sum / precipitation_hours (0.0 when hours == 0)
    final_df["precip_intensity"] = np.where(
        final_df["precipitation_hours"] > 0,
        (final_df["precipitation_sum"] / final_df["precipitation_hours"]).round(3),
        0.0,
    )

    # 2. Hemisphere-adjusted season (0=Winter, 1=Spring, 2=Summer, 3=Autumn)
    month = pd.to_datetime(final_df["date"]).dt.month
    north_season_map = {
        12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3
    }
    south_season_map = {
        12: 2, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 0, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1
    }
    is_south = final_df["location"].str.lower() == "sydney"
    final_df["season"] = np.where(
        is_south, month.map(south_season_map), month.map(north_season_map)
    ).astype(int)

    # 4.1 Feature Selection (Drop the 2 redundant r=0.99 columns)
    cols_to_drop = ["rain_sum", "apparent_temperature_mean"]
    final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], inplace=True)

    # Ensure Target 'RainTomorrow' is the final column
    feature_order = [c for c in final_df.columns if c != "RainTomorrow"] + ["RainTomorrow"]
    final_df = final_df[feature_order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(
        "Saved final processed dataset to %s (%d rows, %d columns)",
        output_path,
        len(final_df),
        len(final_df.columns),
    )
    return final_df


def main() -> None:
    logging.info("Starting sensing pipeline...")

    # 1. Read JSONs & merge
    combined_df = load_raw_json_files(RAW_DATA_DIR, LOCATIONS)

    # 2. Save sensed_data.csv into data/raw/
    sensed_csv_path = RAW_DATA_DIR / "sensed_data.csv"
    sensed_df = create_sensed_data(combined_df, sensed_csv_path)

    # 3. Create RainTomorrow target, drop last day per location, save to data/processed/
    final_csv_path = PROCESSED_DATA_DIR / "final_dataset.csv"
    final_df = create_final_dataset(sensed_df, final_csv_path)

    logging.info("Sensing pipeline finished successfully.")

    # Print summary of target class distribution
    rain_counts = final_df["RainTomorrow"].value_counts()
    rain_pct = final_df["RainTomorrow"].value_counts(normalize=True) * 100
    logging.info(
        "Target distribution:\nNo Rain (0): %d (%.1f%%)\nRain (1): %d (%.1f%%)",
        rain_counts.get(0, 0),
        rain_pct.get(0, 0.0),
        rain_counts.get(1, 0),
        rain_pct.get(1, 0.0),
    )


if __name__ == "__main__":
    main()
