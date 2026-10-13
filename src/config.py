"""Configuration manager reading from params.yaml."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Root directory of the repository
REPO_ROOT = Path(__file__).resolve().parent.parent


def get_repo_root() -> Path:
    """Returns the absolute path to the repository root directory."""
    return REPO_ROOT


def load_params(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads and validates parameters from params.yaml.

    Args:
        config_path: Path to the YAML configuration file. Defaults to params.yaml in REPO_ROOT.

    Returns:
        Dictionary containing project parameters.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If parsing fails.
    """
    if config_path is None:
        config_path = REPO_ROOT / "params.yaml"

    if not config_path.is_file():
        logger.error("Configuration file not found at: %s", config_path)
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            params = yaml.safe_load(f)
            logger.debug("Successfully loaded configuration from: %s", config_path)
            return params or {}
        except yaml.YAMLError as e:
            logger.error("Error parsing YAML config file %s: %s", config_path, e)
            raise


class AppConfig:
    """Convenience wrapper around project parameters."""

    def __init__(self, config_path: Optional[Path] = None):
        self._raw_params = load_params(config_path)

    @property
    def seed(self) -> int:
        return int(self._raw_params.get("seed", 42))

    @property
    def raw_data_path(self) -> Path:
        raw_rel = self._raw_params.get("data", {}).get("raw_path", "data/raw/city_day.csv")
        return REPO_ROOT / raw_rel

    @property
    def clean_params(self) -> Dict[str, Any]:
        return self._raw_params.get("clean", {})

    @property
    def features_params(self) -> Dict[str, Any]:
        return self._raw_params.get("features", {})

    @property
    def split_params(self) -> Dict[str, Any]:
        return self._raw_params.get("split", {})

    @property
    def train_params(self) -> Dict[str, Any]:
        return self._raw_params.get("train", {})

    @property
    def raw_dict(self) -> Dict[str, Any]:
        return self._raw_params
