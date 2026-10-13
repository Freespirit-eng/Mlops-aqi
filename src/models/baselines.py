"""Naive and statistical baseline models for Air Quality forecasting."""

from typing import Any, Dict
import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Predicts next-day AQI as current day's AQI: y_hat_{t+1} = y_t."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns AQI at lag 0 (current day)."""
        if "AQI" in X.columns:
            return X["AQI"].to_numpy()
        elif "AQI_lag0" in X.columns:
            return X["AQI_lag0"].to_numpy()
        raise KeyError("Neither 'AQI' nor 'AQI_lag0' present in feature matrix.")


class SeasonalNaiveBaseline:
    """Predicts next-day AQI as the AQI observed on the same day in the previous week (lag 7)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "AQI_lag7" in X.columns:
            return X["AQI_lag7"].to_numpy()
        raise KeyError("'AQI_lag7' not present in feature matrix.")


class CityMonthMeanBaseline:
    """Predicts next-day AQI as the historical training mean for the corresponding (City, Month)."""

    def __init__(self) -> None:
        self.lookup_table: Dict[Any, float] = {}
        self.global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CityMonthMeanBaseline":
        self.global_mean = float(y.mean())
        df = X.copy()
        df["_target"] = y
        if "City" in df.columns and "Month" in df.columns:
            self.lookup_table = df.groupby(["City", "Month"])["_target"].mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            key = (row.get("City"), row.get("Month"))
            preds.append(self.lookup_table.get(key, self.global_mean))
        return np.array(preds)


class MajorityClassBaseline:
    """Predicts the most frequent AQI category bucket observed in the training split."""

    def __init__(self) -> None:
        self.majority_class: str = "Moderate"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityClassBaseline":
        self.majority_class = str(y.mode().iloc[0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.majority_class] * len(X))
