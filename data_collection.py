"""
Data Collection Script for ML Project Part A.

Fetches historical daily weather data from the Open-Meteo Archive API
for 6 specified cities across required variables, following the project
rate-limiting, error handling (HTTP 429/5xx retry, 400 validation),
and local caching rules.
"""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_DIR = Path("data/raw")

# Rate limiting boundaries from assignment instructions
HOURLY_LIMIT = 5_000
DAILY_LIMIT = 10_000

# 6 mandatory locations with coordinates (lat, lon)
LOCATIONS: Dict[str, Tuple[float, float]] = {
    "london": (51.5074, -0.1278),
    "berlin": (52.5200, 13.4050),
    "new_york": (40.7128, -74.0060),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
    "vancouver": (49.2827, -123.1207),
}

# 10 mandatory daily variables from the project specification
DAILY_VARIABLES: List[str] = [
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

# 20 full calendar years (2006-2025) to span multiple complete ENSO (El Niño / La Niña) cycles
# (~7,305 daily records per city = 43,830 rows total across 6 cities)
START_DATE = "2006-01-01"
END_DATE = "2025-12-31"


def wait_for_rate_limits(hourly_times: deque, daily_times: deque) -> None:
    """Enforce rate limits: max 5,000 req/hr and 10,000 req/24h."""
    while True:
        now = time.time()
        while hourly_times and now - hourly_times[0] >= 3600:
            hourly_times.popleft()
        while daily_times and now - daily_times[0] >= 86400:
            daily_times.popleft()

        waits = []
        if len(hourly_times) >= HOURLY_LIMIT:
            waits.append(3600 - (now - hourly_times[0]))
        if len(daily_times) >= DAILY_LIMIT:
            waits.append(86400 - (now - daily_times[0]))

        if not waits:
            return

        sleep_dur = max(waits) + 1
        logging.warning("Rate limit approached. Sleeping for %.2f seconds.", sleep_dur)
        time.sleep(sleep_dur)


def fetch_with_retries(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = 8,
    backoff_factor: float = 2.0,
) -> dict:
    """
    Fetch data with exponential backoff on HTTP 429 and 5xx errors.
    Fails immediately with clear diagnostic on HTTP 400.
    """
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            response = session.get(url, params=params, timeout=30)
            
            # Explicit check for client error 400
            if response.status_code == 400:
                raise ValueError(
                    f"HTTP 400 Bad Request! Check request parameters: {response.text}"
                )

            # Retry on rate limit 429 or server errors 5xx
            if response.status_code == 429 or 500 <= response.status_code < 600:
                wait_time = backoff_factor ** attempt
                logging.warning(
                    "HTTP %d received. Attempt %d/%d. Waiting %.1fs before retrying...",
                    response.status_code,
                    attempt,
                    max_retries,
                    wait_time,
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            payload = response.json()

            if payload.get("error") is True or "daily" not in payload:
                raise RuntimeError(f"Unexpected response structure: {payload}")

            return payload

        except (requests.ConnectionError, requests.Timeout) as e:
            wait_time = backoff_factor ** attempt
            logging.warning(
                "Network error (%s). Attempt %d/%d. Waiting %.1fs...",
                str(e),
                attempt,
                max_retries,
                wait_time,
            )
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to fetch after {max_retries} attempts.")


def collect(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    force_download: bool = False,
) -> None:
    """
    Collects raw JSON files for all locations from Open-Meteo API.
    Saves results to data/raw/<location>.json.
    Skips downloads if files already exist unless force_download is True.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly_times: deque[float] = deque()
    daily_times: deque[float] = deque()

    logging.info(
        "Starting data collection for %d locations (%s to %s)...",
        len(LOCATIONS),
        start_date,
        end_date,
    )

    with requests.Session() as session:
        for location, (latitude, longitude) in LOCATIONS.items():
            output_file = OUTPUT_DIR / f"{location}.json"

            # Requirement: Do not re-download if file already exists for the target date range
            if output_file.exists() and not force_download:
                try:
                    cached = json.loads(output_file.read_text(encoding="utf-8"))
                    times = cached.get("daily", {}).get("time", [])
                    if times and times[0] == start_date and times[-1] == end_date:
                        logging.info(
                            "[%s] Raw file already exists at %s (%s to %s), skipping download.",
                            location,
                            output_file,
                            start_date,
                            end_date,
                        )
                        continue
                except Exception:
                    pass

            wait_for_rate_limits(hourly_times, daily_times)

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ",".join(DAILY_VARIABLES),
                "timezone": "auto",
            }

            logging.info("[%s] Requesting data...", location)
            payload = fetch_with_retries(session, ENDPOINT, params)

            # Record call times for rate tracking
            now = time.time()
            hourly_times.append(now)
            daily_times.append(now)

            # Save raw unmodified JSON
            output_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logging.info("[%s] Saved successfully to %s", location, output_file)

            # Polite delay between calls
            time.sleep(1.5)

    logging.info("Data collection finished successfully.")


if __name__ == "__main__":
    collect()
