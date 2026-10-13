"""Validation module for raw CPCB Air Quality dataset."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.config import get_repo_root, load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "City",
    "Date",
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
    "AQI",
    "AQI_Bucket",
]

NUMERIC_POLLUTANT_COLUMNS: List[str] = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]


def validate_dataset(
    df: pd.DataFrame,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Runs data quality and schema validation rules on the raw air quality DataFrame.

    Args:
        df: Raw pandas DataFrame loaded from city_day.csv.

    Returns:
        Tuple of:
            - is_valid (bool): True if all mandatory checks pass, False otherwise.
            - metrics (dict): Detailed summary metrics of dataset structure.
            - errors (list[str]): List of critical validation error messages.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Required Columns Check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # If required columns are missing, we cannot safely run downstream checks
    if missing_cols:
        return False, {"missing_columns": missing_cols}, errors

    # 2. Date parsing and future date check
    try:
        parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates_count = int(parsed_dates.isna().sum())
        if invalid_dates_count > 0:
            errors.append(f"Found {invalid_dates_count} dates that failed to parse.")
        
        now = datetime.now()
        future_dates = parsed_dates[parsed_dates > now]
        if not future_dates.empty:
            errors.append(f"Found {len(future_dates)} records with future observation dates.")
    except Exception as e:
        errors.append(f"Date validation exception: {str(e)}")
        parsed_dates = pd.Series(dtype="datetime64[ns]")

    # 3. Duplicate (City, Date) pairs check
    duplicate_mask = df.duplicated(subset=["City", "Date"], keep=False)
    num_duplicates = int(duplicate_mask.sum())
    if num_duplicates > 0:
        warnings.append(f"Found {num_duplicates} duplicate (City, Date) records.")

    # 4. Numeric pollutant values check: Non-negative
    negative_counts: Dict[str, int] = {}
    for col in NUMERIC_POLLUTANT_COLUMNS:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            neg_count = int((numeric_series < 0).sum())
            if neg_count > 0:
                negative_counts[col] = neg_count
                warnings.append(f"Column '{col}' contains {neg_count} negative values.")

    # 5. Missingness statistics
    missingness_pct = {
        col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns
    }

    # 6. City-level statistics: row count and date span
    city_stats: Dict[str, Dict[str, Any]] = {}
    for city, city_group in df.groupby("City"):
        c_dates = pd.to_datetime(city_group["Date"], errors="coerce").dropna()
        city_stats[str(city)] = {
            "total_rows": int(len(city_group)),
            "min_date": str(c_dates.min().strftime("%Y-%m-%d")) if not c_dates.empty else None,
            "max_date": str(c_dates.max().strftime("%Y-%m-%d")) if not c_dates.empty else None,
            "aqi_available_rows": int(city_group["AQI"].notna().sum()),
        }

    overall_metrics: Dict[str, Any] = {
        "validation_timestamp": datetime.now().isoformat(),
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "distinct_cities": int(df["City"].nunique()),
        "date_min": str(parsed_dates.min().strftime("%Y-%m-%d")) if not parsed_dates.empty else None,
        "date_max": str(parsed_dates.max().strftime("%Y-%m-%d")) if not parsed_dates.empty else None,
        "duplicate_city_date_pairs": num_duplicates,
        "negative_pollutant_occurrences": negative_counts,
        "column_missingness_percentage": missingness_pct,
        "target_aqi_missing_percentage": missingness_pct.get("AQI", 0.0),
        "city_breakdown": city_stats,
        "warnings": warnings,
        "errors": errors,
        "passed": len(errors) == 0,
    }

    return len(errors) == 0, overall_metrics, errors


def run_validation(
    data_path: Path,
    output_metrics_path: Path,
) -> bool:
    """Executes dataset validation and writes summary metrics to JSON.

    Args:
        data_path: Path to the raw CSV file.
        output_metrics_path: Path where validation metrics JSON should be saved.

    Returns:
        bool: True if validation succeeded without critical errors, False otherwise.
    """
    logger.info("Starting validation of raw dataset: %s", data_path)

    if not data_path.exists():
        logger.error("Dataset file not found at: %s", data_path)
        output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "passed": False,
                    "error": f"File not found: {data_path}",
                    "validation_timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        return False

    df = pd.read_csv(data_path)
    is_valid, metrics, errors = validate_dataset(df)

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if not is_valid:
        logger.error("Validation failed with %d error(s): %s", len(errors), errors)
    else:
        logger.info(
            "Validation passed successfully! Checked %d rows across %d cities.",
            metrics["total_rows"],
            metrics["distinct_cities"],
        )

    logger.info("Validation metrics written to: %s", output_metrics_path)
    return is_valid


if __name__ == "__main__":
    params = load_params()
    repo_root = get_repo_root()
    raw_rel = params.get("data", {}).get("raw_path", "data/raw/city_day.csv")
    raw_path = repo_root / raw_rel
    metrics_path = repo_root / "reports" / "metrics" / "data_validation.json"

    success = run_validation(raw_path, metrics_path)
    if not success:
        logger.error("Data validation check failed.")
        exit(1)
    logger.info("Data validation pipeline stage complete.")
