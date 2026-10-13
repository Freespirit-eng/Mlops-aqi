"""Data cleaning and imputation pipeline module."""

from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import get_repo_root, load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def clean_data(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Cleans raw CPCB air quality data, handles outliers, imputes missing values, and saves interim parquet.

    Steps (Phase 5):
        1. Parse dates and sort by (City, Date); drop duplicates.
        2. Drop rows missing target (AQI).
        3. Drop cities with fewer than min_rows_per_city.
        4. Replace negative values with NaN.
        5. Winsorize extreme outliers based on train-set percentiles (0.5th/99.5th).
        6. Drop columns exceeding drop_cols_over_missing.
        7. Within each city, forward-fill up to ffill_limit_days; impute remaining with train-set median.
        8. Save clean dataset to data/interim/city_day_clean.parquet.

    Args:
        input_path: Path to raw city_day.csv.
        output_path: Destination path for cleaned parquet file.

    Returns:
        Cleaned pandas DataFrame.
    """
    params = load_params()
    repo_root = get_repo_root()

    if input_path is None:
        input_path = repo_root / params.get("data", {}).get("raw_path", "data/raw/city_day.csv")
    if output_path is None:
        output_path = repo_root / "data" / "interim" / "city_day_clean.parquet"

    logger.info("Cleaning pipeline initialized. Target output: %s", output_path)
    # Full implementation in Phase 5
    raise NotImplementedError("Phase 5 implementation pending. Scaffolding created.")


if __name__ == "__main__":
    clean_data()
