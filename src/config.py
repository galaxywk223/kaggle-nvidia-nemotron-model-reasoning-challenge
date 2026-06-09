from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

COMPETITION_SLUG = "nvidia-nemotron-model-reasoning-challenge"
PUBLIC_ADAPTER_VERSION = (
    "kienngx/nemotron-nano-30b-trained/Transformers/9500s-batch1-lr1e-4/1"
)
PUBLIC_ADAPTER_LABEL = "9500s-batch1-lr1e-4"

DATA_DIR = PROJECT_DIR / "data" / "raw"
ADAPTER_DIR = PROJECT_DIR / "models" / "public_adapter_v1"
SUBMISSION_DIR = PROJECT_DIR / "submissions"
SUBMISSION_ZIP = SUBMISSION_DIR / "submission.zip"
LOG_DIR = PROJECT_DIR / "logs"
SUBMISSION_LOG = LOG_DIR / "submission_history.jsonl"

REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def kaggle_executable() -> str:
    configured = os.environ.get("KAGGLE_EXE")
    if configured:
        return configured

    resolved = shutil.which("kaggle")
    if resolved:
        return resolved

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "Scripts" / "kaggle.exe"
        if candidate.exists():
            return str(candidate)

    default_candidate = Path.home() / ".conda" / "envs" / "Kaggle" / "Scripts" / "kaggle.exe"
    if default_candidate.exists():
        return str(default_candidate)

    return "kaggle"
