from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
MESSAGE = "public hammad structured reasoning sft output"
OUTPUT_ZIP = Path("/kaggle/working/submission.zip")


def find_source_zip() -> Path:
    candidates = sorted(Path("/kaggle/input").rglob("submission.zip"), key=lambda path: len(str(path)))
    if not candidates:
        raise FileNotFoundError("No submission.zip found under /kaggle/input")
    return candidates[0]


def validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        required = {"adapter_config.json", "adapter_model.safetensors"}
        missing = required - names
        if missing:
            raise ValueError(f"Missing required root files: {sorted(missing)}")
        config = json.loads(archive.read("adapter_config.json"))
    if int(config.get("r", 999)) > 32:
        raise ValueError(f"LoRA rank exceeds competition limit: {config.get('r')}")
    print({"zip": str(path), "rank": config.get("r"), "size": path.stat().st_size})


source_zip = find_source_zip()
validate_zip(source_zip)
shutil.copy2(source_zip, OUTPUT_ZIP)
validate_zip(OUTPUT_ZIP)
subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(OUTPUT_ZIP), "-m", MESSAGE],
    check=True,
)
