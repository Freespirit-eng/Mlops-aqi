"""Ingestion stage: verifies existence of raw dataset, invokes validation, and prepares ingestion log."""

import sys
from pathlib import Path

from src.config import get_repo_root, load_params
from src.data.validate import run_validation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def ingest_data() -> None:
    """Verifies raw data availability and runs validation checks."""
    logger.info("Initializing raw data ingestion stage...")
    params = load_params()
    repo_root = get_repo_root()

    raw_rel = params.get("data", {}).get("raw_path", "data/raw/city_day.csv")
    raw_path = repo_root / raw_rel
    metrics_path = repo_root / "reports" / "metrics" / "data_validation.json"

    if not raw_path.exists():
        logger.error(
            "Raw data file not found at expected location: %s. "
            "Please download 'city_day.csv' and place it in 'data/raw/'.",
            raw_path,
        )
        sys.exit(1)

    logger.info("Raw data found at %s. Triggering validation checks...", raw_path)
    is_valid = run_validation(data_path=raw_path, output_metrics_path=metrics_path)

    if not is_valid:
        logger.error("Dataset validation failed. Halting ingestion pipeline.")
        sys.exit(1)

    logger.info("Dataset ingestion and validation successfully completed.")


if __name__ == "__main__":
    ingest_data()
