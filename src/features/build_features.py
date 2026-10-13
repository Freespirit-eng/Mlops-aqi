"""Feature engineering pipeline module: generates temporal lags, rolling aggregations, calendar features, and chronological splits."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import get_repo_root, load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_features(
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates features and outputs leakage-free train, val, and test splits.

    Engineered features (Phase 5):
        - Target: AQI.shift(-1) per city (next-day AQI) and target_bucket
        - Lags: Lags [1, 2, 3, 7] for AQI and key pollutants (PM2.5, PM10, NO2, O3, CO, SO2)
        - Rolling stats: Window [3, 7, 14] mean, std, max for AQI and PM2.5 (shifted by 1)
        - Change features: AQI_diff1, PM25_diff1
        - Calendar & cyclical: Day of week, month, day of year, weekend flag, cyclical sin/cos encodings
        - Categorical: City one-hot encoding or integer encoding
        - Scaling: StandardScaler fitted ONLY on train split; applied to val and test

    Args:
        input_path: Path to interim clean parquet.
        output_dir: Directory where processed train.parquet, val.parquet, test.parquet, and feature_list.json will be saved.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    params = load_params()
    repo_root = get_repo_root()

    if input_path is None:
        input_path = repo_root / "data" / "interim" / "city_day_clean.parquet"
    if output_dir is None:
        output_dir = repo_root / "data" / "processed"

    logger.info("Feature engineering pipeline initialized. Output dir: %s", output_dir)
    # Full implementation in Phase 5
    raise NotImplementedError("Phase 5 feature engineering pending. Scaffolding created.")


if __name__ == "__main__":
    build_features()
