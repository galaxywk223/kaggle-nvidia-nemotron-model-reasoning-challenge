from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


TRAIN_ON_KAGGLE = int(os.environ.get("TRAIN_ON_KAGGLE", "1"))
USE_PRETRAINED = int(os.environ.get("USE_PRETRAINED", "0"))

assert (TRAIN_ON_KAGGLE + USE_PRETRAINED) == 1, "Set exactly one mode."

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_SOURCE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
PRETRAINED_ADAPTER_DATASET_PATH = "/kaggle/input/datasets/dgxchen/trained-adapter"
TRAIN_DATA_PATH = "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv"
OUTPUT_DIR = Path("/kaggle/working")
ADAPTER_OUTPUT_DIR = OUTPUT_DIR / "sft_adapter"
SUBMISSION_ADAPTER_DIR = OUTPUT_DIR / "submission_adapter"
SUBMISSION_ZIP = OUTPUT_DIR / "submission.zip"


def install_offline_packages() -> None:
    import glob
    import torch

    packages_dir = "/kaggle/input/datasets/mayukh18/nemotron-packages/packages"
    if not torch.cuda.is_available():
        raise RuntimeError("TRAIN_ON_KAGGLE=1 requires a Kaggle GPU runtime.")
    capability = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    print(json.dumps({"gpu": device_name, "cuda_capability": capability}, indent=2))
    if capability < (7, 0):
        raise RuntimeError(
            "This training experiment requires a Kaggle GPU with CUDA capability sm_70 or newer. "
            f"The allocated GPU is {device_name} with capability sm_{capability[0]}{capability[1]}. "
            "The intended high-memory GPU target is NvidiaRtxPro6000."
        )
    if not os.path.isdir(packages_dir):
        raise FileNotFoundError(f"Offline package directory not found: {packages_dir}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-index",
            "--find-links",
            packages_dir,
            "unsloth",
            "trl",
            "peft",
            "transformers",
            "datasets",
            "accelerate",
            "bitsandbytes",
        ],
        check=True,
    )

    def recursive_wheels(pattern: str) -> list[str]:
        return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))

    causal_wheels = recursive_wheels("causal*conv1d*.whl")
    mamba_wheels = recursive_wheels("mamba_ssm-*.whl")
    if causal_wheels:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", causal_wheels[-1]], check=True)
    if not mamba_wheels:
        raise FileNotFoundError("No mamba_ssm wheel found under /kaggle/input.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", mamba_wheels[-1]], check=True)


def train_lora_adapter() -> Path:
    import gc
    import math
    import re
    import random
    import time
    from collections import defaultdict

    import unsloth
    import kagglehub
    import pandas as pd
    import torch
    from datasets import Dataset as HFDataset
    from torch.utils.data import DataLoader, Sampler
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    seed = 42
    prompt_suffix = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

    model_path = kagglehub.model_download(MODEL_SOURCE)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=8192,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj",
            "out_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    df = pd.read_csv(TRAIN_DATA_PATH)
    train_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    records = []
    record_types = []
    for _, row in train_df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            continue
        cot_cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
        records.append(
            {
                "messages": [
                    {"role": "user", "content": prompt + prompt_suffix},
                    {"role": "assistant", "content": cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"},
                ]
            }
        )
        record_types.append(str(row["type"]))

    dataset = HFDataset.from_list(records)

    def formatting_prompts_func(example):
        messages = example["messages"]
        conversations = [messages] if messages and isinstance(messages[0], dict) else messages
        texts = []
        for conversation in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return texts

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "sft_output"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_length=8192,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1e9,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=seed,
        report_to="none",
        packing=False,
    )

    def build_stratified_index_order(labels: list[str], batch_size: int, sampler_seed: int) -> list[int]:
        by_label: dict[str, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            by_label[label].append(idx)

        rng = random.Random(sampler_seed)
        for idx_list in by_label.values():
            rng.shuffle(idx_list)

        n_batches = max(1, math.ceil(len(labels) / batch_size))
        batches: list[list[int]] = [[] for _ in range(n_batches)]
        batch_order = list(range(n_batches))
        rng.shuffle(batch_order)

        assigned = 0
        for label in sorted(by_label):
            for idx in by_label[label]:
                batches[batch_order[assigned % n_batches]].append(idx)
                assigned += 1

        order = [idx for batch in batches for idx in batch]
        if len(order) != len(labels):
            raise ValueError("Stratified order size mismatch")
        return order

    class PrecomputedOrderSampler(Sampler[int]):
        def __init__(self, order: list[int]) -> None:
            self.order = list(order)

        def __iter__(self):
            return iter(self.order)

        def __len__(self) -> int:
            return len(self.order)

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *args, stratified_order: list[int] | None = None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.stratified_order = stratified_order

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer requires a train_dataset.")
            if self.stratified_order is None:
                return super().get_train_dataloader()
            if len(self.stratified_order) != len(self.train_dataset):
                raise ValueError("Stratified order length does not match train dataset")

            dataloader_kwargs = {
                "batch_size": self.args.per_device_train_batch_size,
                "sampler": PrecomputedOrderSampler(self.stratified_order),
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "drop_last": self.args.dataloader_drop_last,
            }
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor

            return DataLoader(self.train_dataset, **dataloader_kwargs)

    effective_batch_size = max(
        1,
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    )
    stratified_order = build_stratified_index_order(record_types, effective_batch_size, seed)
    print(f"Approx stratified effective batch size: {effective_batch_size}")
    print("Stratified batching by type:", dict(sorted(pd.Series(record_types).value_counts().to_dict().items())))

    try:
        trainer = StratifiedSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            formatting_func=formatting_prompts_func,
            stratified_order=stratified_order,
        )
    except TypeError:
        trainer = StratifiedSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            formatting_func=formatting_prompts_func,
            stratified_order=stratified_order,
        )

    print("Starting SFT training...")
    started_at = time.time()
    trainer.train()
    elapsed = time.time() - started_at
    print(f"Training done in {elapsed / 60:.1f} min")

    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return ADAPTER_OUTPUT_DIR


def package_adapter(src_adapter_dir: Path) -> Path:
    required = ["adapter_config.json", "adapter_model.safetensors"]
    SUBMISSION_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    for name in required:
        src = src_adapter_dir / name
        dst = SUBMISSION_ADAPTER_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required adapter file: {src}")
        shutil.copy2(src, dst)

    config_path = SUBMISSION_ADAPTER_DIR / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = BASE_MODEL_NAME
    config["inference_mode"] = True
    config["lora_dropout"] = 0.0
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if SUBMISSION_ZIP.exists():
        SUBMISSION_ZIP.unlink()
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name in required:
            archive.write(SUBMISSION_ADAPTER_DIR / name, arcname=name)

    with zipfile.ZipFile(SUBMISSION_ZIP) as archive:
        names = set(archive.namelist())
    if set(required) != names:
        raise RuntimeError(f"Unexpected submission.zip layout: {sorted(names)}")
    return SUBMISSION_ZIP


def main() -> int:
    print(
        json.dumps(
            {
                "TRAIN_ON_KAGGLE": TRAIN_ON_KAGGLE,
                "USE_PRETRAINED": USE_PRETRAINED,
                "BASE_MODEL_NAME": BASE_MODEL_NAME,
                "TRAIN_DATA_PATH": TRAIN_DATA_PATH,
            },
            indent=2,
        )
    )
    if TRAIN_ON_KAGGLE:
        install_offline_packages()
        adapter_dir = train_lora_adapter()
    else:
        adapter_dir = Path(PRETRAINED_ADAPTER_DATASET_PATH)

    submission_zip = package_adapter(adapter_dir)
    print(json.dumps({"submission_zip": str(submission_zip), "size_bytes": submission_zip.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
