from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRAIN_ON_KAGGLE = int(os.environ.get("TRAIN_ON_KAGGLE", "1"))
USE_PRETRAINED = int(os.environ.get("USE_PRETRAINED", "0"))

assert (TRAIN_ON_KAGGLE + USE_PRETRAINED) == 1, "Set exactly one mode."

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_SOURCE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
PRETRAINED_ADAPTER_DATASET_PATH = "/kaggle/input/datasets/dgxchen/trained-adapter"
# Matched prompt-answer-CoT records produced by the Teacher-CoT construction stage.
TRAIN_DATA_PATH = "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv"
OUTPUT_DIR = Path("/kaggle/working")
ADAPTER_OUTPUT_DIR = OUTPUT_DIR / "sft_adapter"
SUBMISSION_ADAPTER_DIR = OUTPUT_DIR / "submission_adapter"
SUBMISSION_ZIP = OUTPUT_DIR / "submission.zip"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class PreparedSFTExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    loss_weights: list[float]
    sample_id: str
    problem_type: str
    truncated: bool
    supervised_tokens: int
    answer_weighted_tokens: int

    def to_record(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "loss_weights": self.loss_weights,
            "id": self.sample_id,
            "type": self.problem_type,
            "truncated": self.truncated,
        }


def strip_boxed_answers(text: str) -> str:
    import re

    return re.sub(r"\\boxed\{[^}]*\}", "", text).rstrip()


def render_chat_text(tokenizer: Any, prompt: str, assistant_content: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def assert_tokenizer_supports_offsets(tokenizer: Any) -> None:
    try:
        encoded = tokenizer("offset smoke test", return_offsets_mapping=True, add_special_tokens=False)
    except Exception as exc:  # pragma: no cover - exception type depends on tokenizer implementation.
        raise RuntimeError(
            "The training tokenizer must support return_offsets_mapping=True. "
            "Use a fast tokenizer or a tokenizer implementation that returns offsets."
        ) from exc
    if "offset_mapping" not in encoded:
        raise RuntimeError("The training tokenizer did not return offset_mapping.")
    if len(encoded["offset_mapping"]) != len(encoded["input_ids"]):
        raise RuntimeError("Tokenizer offset_mapping length does not match input_ids length.")


def _find_unique_span(text: str, needle: str, label: str, sample_id: str) -> tuple[int, int]:
    start = text.find(needle)
    if start < 0:
        raise ValueError(f"Sample {sample_id}: {label} was not found in the rendered chat text.")
    next_start = text.find(needle, start + len(needle))
    if next_start >= 0:
        raise ValueError(f"Sample {sample_id}: {label} is not unique in the rendered chat text.")
    return start, start + len(needle)


def _tokenize_with_offsets(tokenizer: Any, rendered_text: str, sample_id: str) -> dict[str, Any]:
    try:
        encoded = tokenizer(
            rendered_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Sample {sample_id}: tokenizer failed with return_offsets_mapping=True."
        ) from exc
    if "offset_mapping" not in encoded:
        raise RuntimeError(f"Sample {sample_id}: tokenizer did not return offset_mapping.")
    if len(encoded["offset_mapping"]) != len(encoded["input_ids"]):
        raise RuntimeError(f"Sample {sample_id}: offset_mapping length does not match input_ids length.")
    return encoded


def _overlaps_span(token_span: tuple[int, int] | list[int], char_span: tuple[int, int]) -> bool:
    token_start, token_end = int(token_span[0]), int(token_span[1])
    char_start, char_end = char_span
    if token_end <= token_start:
        return False
    return token_start < char_end and token_end > char_start


def _build_tokenized_example(
    *,
    tokenizer: Any,
    prompt: str,
    cot_text: str,
    answer: str,
    sample_id: str,
    problem_type: str,
    answer_weight: float,
    truncated: bool,
) -> PreparedSFTExample:
    answer_marker = f"</think>\n\\boxed{{{answer}}}"
    assistant_content = cot_text + "\n" + answer_marker
    rendered_text = render_chat_text(tokenizer, prompt, assistant_content)
    assistant_span = _find_unique_span(rendered_text, assistant_content, "assistant content", sample_id)
    answer_span = _find_unique_span(rendered_text, answer_marker, "answer suffix", sample_id)
    encoded = _tokenize_with_offsets(tokenizer, rendered_text, sample_id)

    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    labels: list[int] = []
    loss_weights: list[float] = []
    supervised_tokens = 0
    answer_weighted_tokens = 0

    for token_id, offset in zip(input_ids, encoded["offset_mapping"], strict=True):
        is_assistant = _overlaps_span(offset, assistant_span)
        is_answer = _overlaps_span(offset, answer_span)
        if is_assistant:
            labels.append(int(token_id))
            supervised_tokens += 1
            if is_answer:
                loss_weights.append(float(answer_weight))
                answer_weighted_tokens += 1
            else:
                loss_weights.append(1.0)
        else:
            labels.append(IGNORE_INDEX)
            loss_weights.append(0.0)

    if supervised_tokens == 0:
        raise ValueError(f"Sample {sample_id}: no assistant tokens were supervised.")
    if answer_weighted_tokens == 0:
        raise ValueError(f"Sample {sample_id}: answer suffix did not cover any supervised tokens.")

    return PreparedSFTExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
        sample_id=sample_id,
        problem_type=problem_type,
        truncated=truncated,
        supervised_tokens=supervised_tokens,
        answer_weighted_tokens=answer_weighted_tokens,
    )


def prepare_sft_example(
    *,
    tokenizer: Any,
    prompt: str,
    answer: str,
    cot_text: str,
    sample_id: str,
    problem_type: str,
    max_length: int,
    answer_weight: float = 2.0,
) -> PreparedSFTExample:
    full_example = _build_tokenized_example(
        tokenizer=tokenizer,
        prompt=prompt,
        cot_text=cot_text,
        answer=answer,
        sample_id=sample_id,
        problem_type=problem_type,
        answer_weight=answer_weight,
        truncated=False,
    )
    if len(full_example.input_ids) <= max_length:
        return full_example

    empty_example = _build_tokenized_example(
        tokenizer=tokenizer,
        prompt=prompt,
        cot_text="",
        answer=answer,
        sample_id=sample_id,
        problem_type=problem_type,
        answer_weight=answer_weight,
        truncated=True,
    )
    if len(empty_example.input_ids) > max_length:
        raise ValueError(
            f"Sample {sample_id}: prompt plus required answer suffix uses {len(empty_example.input_ids)} "
            f"tokens, exceeding max_length={max_length}."
        )

    low = 0
    high = len(cot_text)
    best = empty_example
    while low <= high:
        mid = (low + high) // 2
        candidate_cot = cot_text[:mid].rstrip()
        candidate = _build_tokenized_example(
            tokenizer=tokenizer,
            prompt=prompt,
            cot_text=candidate_cot,
            answer=answer,
            sample_id=sample_id,
            problem_type=problem_type,
            answer_weight=answer_weight,
            truncated=True,
        )
        if len(candidate.input_ids) <= max_length:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    return best


class WeightedDataCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        label_pad_token_id: int = IGNORE_INDEX,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weights": [],
        }
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            batch["labels"].append(feature["labels"] + [self.label_pad_token_id] * pad_length)
            batch["loss_weights"].append(feature["loss_weights"] + [0.0] * pad_length)

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float32),
        }


def weighted_causal_lm_loss(logits: Any, labels: Any, loss_weights: Any) -> Any:
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].to(device=shift_logits.device).contiguous()

    active = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~active, 0)
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    shift_weights = shift_weights.to(dtype=token_loss.dtype).masked_fill(~active, 0.0)
    return (token_loss * shift_weights).sum() / shift_weights.sum().clamp_min(1.0)


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
    import random
    import time
    from collections import defaultdict

    import unsloth
    import kagglehub
    import pandas as pd
    import torch
    from datasets import Dataset as HFDataset
    from torch.utils.data import DataLoader, Sampler
    from transformers import Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    seed = 42
    max_length = 8192
    answer_weight = 2.0
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
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer must define pad_token_id or eos_token_id for padded training batches.")
    assert_tokenizer_supports_offsets(tokenizer)

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
    skipped_short_cot = 0
    truncated_count = 0
    supervised_token_count = 0
    answer_weighted_token_count = 0
    for _, row in train_df.iterrows():
        sample_id = str(row.get("id", len(records)))
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            skipped_short_cot += 1
            continue
        cot_cleaned = strip_boxed_answers(cot)
        prepared = prepare_sft_example(
            tokenizer=tokenizer,
            prompt=prompt + prompt_suffix,
            answer=answer,
            cot_text=cot_cleaned,
            sample_id=sample_id,
            problem_type=str(row["type"]),
            max_length=max_length,
            answer_weight=answer_weight,
        )
        records.append(prepared.to_record())
        record_types.append(prepared.problem_type)
        truncated_count += int(prepared.truncated)
        supervised_token_count += prepared.supervised_tokens
        answer_weighted_token_count += prepared.answer_weighted_tokens

    if not records:
        raise RuntimeError("No SFT records were prepared for training.")
    dataset = HFDataset.from_list(records)
    print(
        json.dumps(
            {
                "prepared_records": len(records),
                "skipped_short_cot": skipped_short_cot,
                "truncated_records": truncated_count,
                "supervised_tokens": supervised_token_count,
                "answer_weighted_tokens": answer_weighted_token_count,
                "answer_weight": answer_weight,
                "max_length": max_length,
            },
            indent=2,
        )
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "sft_output"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
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

    class WeightedCausalLMTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            loss_weights = inputs.pop("loss_weights")
            outputs = model(**inputs)
            loss = weighted_causal_lm_loss(outputs.logits, labels, loss_weights)
            return (loss, outputs) if return_outputs else loss

    class StratifiedTrainer(WeightedCausalLMTrainer):
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
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": WeightedDataCollator(pad_token_id=tokenizer.pad_token_id),
        "stratified_order": stratified_order,
    }
    try:
        trainer = StratifiedTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = StratifiedTrainer(**trainer_kwargs, tokenizer=tokenizer)

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
