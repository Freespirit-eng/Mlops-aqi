"""AQI utilities: CPCB scale category mapping, integer encoding, and health advisories."""

from typing import Dict, Optional, Union
import numpy as np

# CPCB official categorical buckets ordered by severity
AQI_BUCKETS = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
]

# Integer mappings for classification targets
BUCKET_TO_INT: Dict[str, int] = {bucket: idx for idx, bucket in enumerate(AQI_BUCKETS)}
INT_TO_BUCKET: Dict[int, str] = {idx: bucket for idx, bucket in enumerate(AQI_BUCKETS)}

# Plain-language health guidance for the Air Quality Awareness Assistant
HEALTH_ADVICE: Dict[str, str] = {
    "Good": "Air quality is ideal; enjoy outdoor activities with no health restrictions.",
    "Satisfactory": "Air quality is acceptable; unusually sensitive individuals should observe mild caution.",
    "Moderate": "Air quality is acceptable, but individuals with asthma or heart conditions may experience mild discomfort.",
    "Poor": "Air quality is unhealthy; sensitive groups should limit outdoor exertion and wear masks.",
    "Very Poor": "Air quality is very unhealthy; everyone should avoid prolonged outdoor exposure and use air purifiers.",
    "Severe": "Air quality is hazardous; avoid all outdoor activity, keep windows sealed, and consult medical professionals if symptomatic.",
}


def aqi_to_bucket(aqi: Union[float, int]) -> str:
    """Maps a continuous numeric AQI value to the official Indian CPCB category bucket.

    Thresholds:
        0 - 50:    Good
        51 - 100:  Satisfactory
        101 - 200: Moderate
        201 - 300: Poor
        301 - 400: Very Poor
        401+:      Severe

    Args:
        aqi: Numeric AQI value (float or int).

    Returns:
        String representing the AQI category.

    Raises:
        ValueError: If aqi is negative or NaN.
    """
    if aqi is None or (isinstance(aqi, (float, np.floating)) and np.isnan(aqi)):
        raise ValueError("AQI value cannot be NaN or None.")

    if aqi < 0:
        raise ValueError(f"AQI value cannot be negative: {aqi}")

    # CPCB integer/floating threshold boundaries
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def bucket_to_int(bucket: str) -> int:
    """Maps an AQI bucket name to its integer class index (0 to 5).

    Args:
        bucket: Category name string.

    Returns:
        Integer class index.

    Raises:
        ValueError: If bucket name is invalid.
    """
    if bucket not in BUCKET_TO_INT:
        raise ValueError(f"Unknown AQI bucket: '{bucket}'. Expected one of {list(BUCKET_TO_INT.keys())}")
    return BUCKET_TO_INT[bucket]


def int_to_bucket(index: int) -> str:
    """Maps an integer class index (0 to 5) back to the AQI bucket name.

    Args:
        index: Class integer index.

    Returns:
        AQI category string.

    Raises:
        ValueError: If index is outside 0 to 5.
    """
    if index not in INT_TO_BUCKET:
        raise ValueError(f"Invalid bucket index: {index}. Expected 0 to {len(AQI_BUCKETS) - 1}")
    return INT_TO_BUCKET[index]


def get_health_advice(bucket: str) -> str:
    """Retrieves standard health guidance advice for a given category bucket.

    Args:
        bucket: AQI bucket name.

    Returns:
        Plain-language health advisory sentence.
    """
    if bucket not in HEALTH_ADVICE:
        raise ValueError(f"Unknown bucket '{bucket}'. Expected one of {list(HEALTH_ADVICE.keys())}")
    return HEALTH_ADVICE[bucket]
