"""Unit tests for AQI utility functions and CPCB bucket mappings."""

import numpy as np
import pytest

from src.features.aqi_utils import (
    AQI_BUCKETS,
    HEALTH_ADVICE,
    aqi_to_bucket,
    bucket_to_int,
    get_health_advice,
    int_to_bucket,
)


@pytest.mark.parametrize(
    "aqi, expected_bucket",
    [
        (0.0, "Good"),
        (25.0, "Good"),
        (50.0, "Good"),
        (50.1, "Satisfactory"),
        (51.0, "Satisfactory"),
        (75.0, "Satisfactory"),
        (100.0, "Satisfactory"),
        (100.1, "Moderate"),
        (101.0, "Moderate"),
        (150.0, "Moderate"),
        (200.0, "Moderate"),
        (200.1, "Poor"),
        (201.0, "Poor"),
        (250.0, "Poor"),
        (300.0, "Poor"),
        (301.0, "Very Poor"),
        (350.0, "Very Poor"),
        (400.0, "Very Poor"),
        (400.1, "Severe"),
        (401.0, "Severe"),
        (500.0, "Severe"),
        (750.0, "Severe"),
    ],
)
def test_aqi_to_bucket_boundaries(aqi: float, expected_bucket: str) -> None:
    """Tests CPCB AQI thresholds at standard values and exact boundary transitions."""
    assert aqi_to_bucket(aqi) == expected_bucket


@pytest.mark.parametrize("invalid_aqi", [-1.0, -0.01, -100.0])
def test_aqi_to_bucket_negative_raises_value_error(invalid_aqi: float) -> None:
    """Negative AQI values are physically impossible and must raise ValueError."""
    with pytest.raises(ValueError, match="cannot be negative"):
        aqi_to_bucket(invalid_aqi)


def test_aqi_to_bucket_nan_raises_value_error() -> None:
    """NaN or None values must raise ValueError."""
    with pytest.raises(ValueError, match="cannot be NaN or None"):
        aqi_to_bucket(np.nan)


def test_bucket_int_round_trip() -> None:
    """Verifies that integer mapping and reverse mapping are bijective."""
    for bucket in AQI_BUCKETS:
        idx = bucket_to_int(bucket)
        assert isinstance(idx, int)
        assert 0 <= idx < len(AQI_BUCKETS)
        assert int_to_bucket(idx) == bucket


def test_bucket_to_int_invalid() -> None:
    """Invalid bucket name must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown AQI bucket"):
        bucket_to_int("SuperHazardous")


def test_int_to_bucket_out_of_range() -> None:
    """Index outside 0-5 must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid bucket index"):
        int_to_bucket(99)
    with pytest.raises(ValueError, match="Invalid bucket index"):
        int_to_bucket(-1)


def test_health_advice_all_buckets_covered() -> None:
    """Verifies that all 6 CPCB buckets have descriptive health advice."""
    for bucket in AQI_BUCKETS:
        advice = get_health_advice(bucket)
        assert isinstance(advice, str)
        assert len(advice) > 20
        assert advice == HEALTH_ADVICE[bucket]
