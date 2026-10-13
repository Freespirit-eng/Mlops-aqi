"""Training pipeline module: fits regression and classification baselines with MLflow tracking."""

from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import get_repo_root, load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_models(
    processed_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> None:
    """Trains regression and classification baselines and logs metrics and artifacts to MLflow.

    Workflow (Phase 6 & 7):
        1. Load train.parquet, val.parquet, feature_list.json.
        2. Set MLflow tracking URI (sqlite:///mlflow.db).
        3. Fit Ridge, Random Forest, XGBoost, LightGBM models.
        4. Log hyperparameters, metrics (RMSE, MAE, R2, macro-F1), artifacts.
        5. Output val_metrics.json and serialize model artifacts to models/.

    Args:
        processed_dir: Path to directory containing processed parquets.
        models_dir: Path to directory where trained models will be saved.
    """
    params = load_params()
    repo_root = get_repo_root()

    if processed_dir is None:
        processed_dir = repo_root / "data" / "processed"
    if models_dir is None:
        models_dir = repo_root / "models"

    logger.info("Model training pipeline initialized. Models directory: %s", models_dir)
    # Full implementation in Phase 6
    raise NotImplementedError("Phase 6 training implementation pending. Scaffolding created.")


if __name__ == "__main__":
    train_models()
