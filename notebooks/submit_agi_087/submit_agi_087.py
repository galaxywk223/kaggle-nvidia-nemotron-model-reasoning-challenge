from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
MESSAGE = "public agi 0.87 svd adapter kernel-output"
WORKING = Path("/kaggle/working")
OUTPUT_ZIP = WORKING / "submission.zip"


def find_source_zip() -> Path:
    input_root = Path("/kaggle/input")
    candidates = sorted(
        input_root.rglob("submission.zip"),
        key=lambda path: ("agi-for-medal" not in str(path).lower(), len(str(path))),
    )
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
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Unexpected peft_type: {config.get('peft_type')}")
    if int(config.get("r", 999)) > 32:
        raise ValueError(f"LoRA rank exceeds competition limit: {config.get('r')}")
    print({"source_zip": str(path), "rank": config.get("r"), "size": path.stat().st_size})


source_zip = find_source_zip()
validate_zip(source_zip)
shutil.copy2(source_zip, OUTPUT_ZIP)
validate_zip(OUTPUT_ZIP)

command = [
    "kaggle",
    "competitions",
    "submit",
    "-c",
    COMPETITION,
    "-f",
    str(OUTPUT_ZIP),
    "-m",
    MESSAGE,
]
print(" ".join(command))
subprocess.run(command, check=True)
