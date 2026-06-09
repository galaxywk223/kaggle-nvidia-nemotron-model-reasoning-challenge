from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
MESSAGE = "public safar 0.86 repacked adapter kernel-output"
OUTPUT_ZIP = Path("/kaggle/working/submission.zip")


def find_adapter_dir() -> Path:
    candidates = []
    for config_path in Path("/kaggle/input").rglob("adapter_config.json"):
        adapter_dir = config_path.parent
        if (adapter_dir / "adapter_model.safetensors").exists():
            candidates.append(adapter_dir)
    if not candidates:
        raise FileNotFoundError("No adapter_config.json + adapter_model.safetensors pair found")
    candidates.sort(key=lambda p: ("lb-score" not in str(p).lower(), len(str(p))))
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
    print({"rank": config.get("r"), "size": path.stat().st_size})


adapter_dir = find_adapter_dir()
print({"adapter_dir": str(adapter_dir)})
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        archive.write(adapter_dir / name, name)
validate_zip(OUTPUT_ZIP)
subprocess.run(
    [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(OUTPUT_ZIP),
        "-m",
        MESSAGE,
    ],
    check=True,
)
