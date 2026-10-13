"""Unit tests for dataset validation logic in src/data/validate.py."""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from src.data.validate import REQUIRED_COLUMNS, validate_dataset


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Creates a minimal valid DataFrame matching the CPCB schema."""
    data = {col: [10.0, 20.0] for col in REQUIRED_COLUMNS}
    data["City"] = ["Bengaluru", "Bengaluru"]
    data["Date"] = ["2018-01-01", "2018-01-02"]
    data["AQI"] = [85.0, 92.0]
    data["AQI_Bucket"] = ["Satisfactory", "Satisfactory"]
    return pd.DataFrame(data)


def test_validate_dataset_success(sample_valid_df: pd.DataFrame) -> None:
    """Verifies that a well-formed DataFrame passes validation with zero errors."""
    is_valid, metrics, errors = validate_dataset(sample_valid_df)
    assert is_valid is True
    assert len(errors) == 0
    assert metrics["total_rows"] == 2
    assert metrics["distinct_cities"] == 1
    assert metrics["passed"] is True


def test_validate_dataset_missing_column(sample_valid_df: pd.DataFrame) -> None:
    """Verifies that dropping a required column causes validation to fail."""
    df_missing = sample_valid_df.drop(columns=["PM2.5"])
    is_valid, metrics, errors = validate_dataset(df_missing)
    assert is_valid is False
    assert any("PM2.5" in err for err in errors)


def test_validate_dataset_future_date(sample_valid_df: pd.DataFrame) -> None:
    """Verifies that observation dates in the future trigger validation failure."""
    future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    df_future = sample_valid_df.copy()
    df_future.loc[0, "Date"] = future_date
    is_valid, metrics, errors = validate_dataset(df_future)
    assert is_valid is False
    assert any("future observation dates" in err for err in errors)


def test_validate_dataset_negative_pollutant_generates_warning(sample_valid_df: pd.DataFrame) -> None:
    """Verifies that negative pollutant concentrations are detected and flagged in warnings."""
    df_neg = sample_valid_df.copy()
    df_neg.loc[0, "PM2.5"] = -15.0
    is_valid, metrics, errors = validate_dataset(df_neg)
    # Negative pollutant is treated as a data quality warning (to be cleaned in clean.py)
    assert metrics["negative_pollutant_occurrences"].get("PM2.5") == 1
    assert any("PM2.5" in warn for warn in metrics["warnings"])
