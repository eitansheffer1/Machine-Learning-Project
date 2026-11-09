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
    final_df = df_sorted[~last_day_mask].reset_index(drop=True)

    logging.info(
        "Created 'RainTomorrow' target. Dropped %d last-day records without future labels.",
        dropped_count,
    )

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
