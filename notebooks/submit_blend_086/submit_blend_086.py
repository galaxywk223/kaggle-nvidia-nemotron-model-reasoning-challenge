from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
MESSAGE = "custom diverse 0.86 adapter tensor average"
WORKING = Path("/kaggle/working")
OUTPUT_ZIP = WORKING / "submission.zip"
WEIGHTS = {
    "mohamed": 0.4,
    "mirza": 0.35,
    "taha": 0.25,
}


def classify_source(path: Path) -> str | None:
    lowered = str(path).lower()
    for name in WEIGHTS:
        if name in lowered:
            return name
    return None


def validate_zip(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        required = {"adapter_config.json", "adapter_model.safetensors"}
        missing = required - names
        if missing:
            raise ValueError(f"Missing required root files in {path}: {sorted(missing)}")
        config = json.loads(archive.read("adapter_config.json"))
    if int(config.get("r", 999)) > 32:
        raise ValueError(f"LoRA rank exceeds competition limit in {path}: {config.get('r')}")
    return config


def collect_source_zips() -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted(Path("/kaggle/input").rglob("submission.zip"), key=lambda item: len(str(item))):
        source = classify_source(path)
        if source is None or source in selected:
            continue
        validate_zip(path)
        selected[source] = path
    missing = sorted(set(WEIGHTS) - set(selected))
    if missing:
        raise FileNotFoundError(f"Missing source zips for: {missing}; found={selected}")
    return selected


def extract_adapter(zip_path: Path, output_dir: Path) -> Path:
    target = output_dir / zip_path.parent.name
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extract("adapter_config.json", target)
        archive.extract("adapter_model.safetensors", target)
    return target


def average_state_dict(adapter_dirs: dict[str, Path]) -> dict[str, torch.Tensor]:
    state_dicts = {
        name: load_file(str(path / "adapter_model.safetensors"), device="cpu")
        for name, path in adapter_dirs.items()
    }
    common_keys = set.intersection(*(set(state) for state in state_dicts.values()))
    if not common_keys:
        raise ValueError("No common tensor keys found across adapters")

    blended = {}
    for key in sorted(common_keys):
        tensors = {name: state[key] for name, state in state_dicts.items()}
        shapes = {tuple(tensor.shape) for tensor in tensors.values()}
        if len(shapes) != 1:
            continue
        accumulator = None
        for name, tensor in tensors.items():
            weighted = tensor.to(torch.float32) * WEIGHTS[name]
            accumulator = weighted if accumulator is None else accumulator + weighted
        blended[key] = accumulator.to(next(iter(tensors.values())).dtype)
    if not blended:
        raise ValueError("No compatible tensor keys found across adapters")
    print({"blended_tensor_count": len(blended), "source_count": len(adapter_dirs)})
    return blended


source_zips = collect_source_zips()
print({"source_zips": {name: str(path) for name, path in source_zips.items()}})
with tempfile.TemporaryDirectory() as temp_name:
    temp_dir = Path(temp_name)
    adapter_dirs = {name: extract_adapter(path, temp_dir) for name, path in source_zips.items()}
    base_config_path = adapter_dirs["mohamed"] / "adapter_config.json"
    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if int(config.get("r", 999)) > 32:
        raise ValueError(f"LoRA rank exceeds competition limit: {config.get('r')}")
    blended_state = average_state_dict(adapter_dirs)
    blended_dir = WORKING / "blended_adapter"
    blended_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config_path, blended_dir / "adapter_config.json")
    save_file(blended_state, str(blended_dir / "adapter_model.safetensors"))

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(WORKING / "blended_adapter" / "adapter_config.json", "adapter_config.json")
    archive.write(WORKING / "blended_adapter" / "adapter_model.safetensors", "adapter_model.safetensors")

validate_zip(OUTPUT_ZIP)
subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(OUTPUT_ZIP), "-m", MESSAGE],
    check=True,
)
