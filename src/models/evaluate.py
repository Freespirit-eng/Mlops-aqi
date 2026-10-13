"""Model evaluation module: evaluates best models on test split, generates diagnostic figures and reports."""

from pathlib import Path
from typing import Optional

from src.config import get_repo_root, load_params
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_models(
    models_dir: Optional[Path] = None,
    test_data_path: Optional[Path] = None,
    output_figures_dir: Optional[Path] = None,
    output_metrics_file: Optional[Path] = None,
) -> None:
    """Evaluates final selected models on test split (2020) and outputs publication figures.

    Tasks (Phase 8):
        1. Load test.parquet (strictly evaluated once).
        2. Compute regression metrics: RMSE, MAE, R2, MAPE, within-20 rate.
        3. Compute classification metrics: macro-F1, balanced accuracy, within-1-bucket accuracy.
        4. Generate error breakdown plots by city, month, and severity.
        5. Generate Bengaluru regional spotlight analysis.
        6. Export test_metrics.json and high-resolution figures.

    Args:
        models_dir: Directory containing serialized model artifacts.
        test_data_path: Path to test.parquet.
        output_figures_dir: Directory for evaluation plots.
        output_metrics_file: JSON destination for test metrics.
    """
    params = load_params()
    repo_root = get_repo_root()

    if models_dir is None:
        models_dir = repo_root / "models"
    if test_data_path is None:
        test_data_path = repo_root / "data" / "processed" / "test.parquet"
    if output_figures_dir is None:
        output_figures_dir = repo_root / "reports" / "figures" / "evaluation"
    if output_metrics_file is None:
        output_metrics_file = repo_root / "reports" / "metrics" / "test_metrics.json"

    logger.info("Evaluation pipeline initialized. Reporting to: %s", output_metrics_file)
    # Full implementation in Phase 8
    raise NotImplementedError("Phase 8 evaluation pending. Scaffolding created.")


if __name__ == "__main__":
    evaluate_models()
