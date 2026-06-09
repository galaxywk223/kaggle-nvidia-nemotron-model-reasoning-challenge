from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
MESSAGE = 'public rauff modular v2 lora output'
OUTPUT_ZIP = Path("/kaggle/working/submission.zip")
REQUIRED = {"adapter_config.json", "adapter_model.safetensors"}


def validate_zip(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        missing = REQUIRED - names
        if missing:
            raise ValueError(f"Missing required root files: {sorted(missing)}")
        config = json.loads(archive.read("adapter_config.json"))
    if int(config.get("r", 999)) > 32:
        raise ValueError(f"LoRA rank exceeds competition limit: {config.get('r')}")
    print({"zip": str(path), "rank": config.get("r"), "size": path.stat().st_size})
    return config


def find_valid_source_zip() -> Path | None:
    candidates = sorted(Path("/kaggle/input").rglob("submission.zip"), key=lambda path: len(str(path)))
    for path in candidates:
        try:
            validate_zip(path)
        except Exception as exc:
            print({"skip_zip": str(path), "reason": str(exc)})
            continue
        return path
    return None


def find_adapter_dir() -> Path:
    candidates = []
    for config_path in Path("/kaggle/input").rglob("adapter_config.json"):
        adapter_dir = config_path.parent
        if not (adapter_dir / "adapter_model.safetensors").exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config.get("r", 999)) > 32:
            print({"skip_adapter_dir": str(adapter_dir), "rank": config.get("r")})
            continue
        candidates.append(adapter_dir)
    if not candidates:
        raise FileNotFoundError("No valid adapter_config.json + adapter_model.safetensors pair found")
    candidates.sort(key=lambda path: (len(str(path)), str(path)))
    return candidates[0]


def package_adapter_dir(adapter_dir: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(REQUIRED):
            archive.write(adapter_dir / name, name)


source_zip = find_valid_source_zip()
if source_zip is not None:
    shutil.copy2(source_zip, OUTPUT_ZIP)
else:
    adapter_dir = find_adapter_dir()
    print({"adapter_dir": str(adapter_dir)})
    package_adapter_dir(adapter_dir, OUTPUT_ZIP)
validate_zip(OUTPUT_ZIP)
subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(OUTPUT_ZIP), "-m", MESSAGE],
    check=True,
)
