from __future__ import annotations

"""Configuration loader for the PD classification pipeline."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project path configuration."""

    root_dir: str = Field(..., alias="root_dir")
    raw_data_dir: str
    processed_dir: str
    features_dir: str
    models_dir: str
    reports_dir: str
    logs_dir: str
    random_state: int
    jobs: int


class RuntimeConfig(BaseModel):
    """Runtime behaviour switches."""

    checkpoint_interval: int
    resume: bool
    verbosity: str
    save_confusion_matrices: bool
    save_roc_curves: bool
    model_registry_format: str


class Config(BaseModel):
    """Root configuration model."""

    project: ProjectConfig
    runtime: RuntimeConfig


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path = CONFIG_PATH) -> Config:
    """Load and validate ``config.yaml``.

    Parameters
    ----------
    path
        Path to the configuration YAML file.

    Returns
    -------
    Config
        Parsed configuration instance.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return Config(**data)


CONFIG = load_config(CONFIG_PATH)

ROOT_DIR = Path(CONFIG.project.root_dir).resolve()
RAW_DIR = ROOT_DIR / CONFIG.project.raw_data_dir
PROCESSED_DIR = ROOT_DIR / CONFIG.project.processed_dir
FEATURES_DIR = ROOT_DIR / CONFIG.project.features_dir
MODELS_DIR = ROOT_DIR / CONFIG.project.models_dir
REPORTS_DIR = ROOT_DIR / CONFIG.project.reports_dir
LOGS_DIR = ROOT_DIR / CONFIG.project.logs_dir
