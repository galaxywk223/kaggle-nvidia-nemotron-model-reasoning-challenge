from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from src.config import DATA_DIR, PROJECT_DIR


DEFAULT_INPUT = DATA_DIR / "train.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "derived" / "problem_ids_matched.csv"
DEFAULT_CONFIG = PROJECT_DIR / "teacher_cot.local.json"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_PROBE_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
PROBE_PROMPT = "What is 2 + 3?"
PROBE_ANSWER = "5"
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
NON_TEXT_MODEL_FRAGMENTS = (
    "audio",
    "realtime",
    "image",
    "codex",
    "embedding",
    "tts",
    "transcribe",
    "speech",
)
PREFERRED_TEXT_MODELS = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.2-pro",
    "gpt-5.2",
    "gpt-4.1",
    "gpt-4o",
)

MATCHED_FIELDS = ("id", "prompt", "answer", "generated_cot", "type")
REJECTED_FIELDS = (
    "id",
    "prompt",
    "answer",
    "generated_cot",
    "type",
    "teacher_answer",
    "match",
    "error",
)


class TeacherProvider(Protocol):
    def generate(self, prompt: str, expected_answer: str) -> dict[str, str]:
        ...


@dataclass(frozen=True)
class OpenAICompatibleTeacher:
    base_url: str
    model: str
    api_key: str
    temperature: float
    max_output_tokens: int
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS

    def generate(self, prompt: str, expected_answer: str) -> dict[str, str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(prompt, expected_answer)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    body = response.read().decode("utf-8")
                content = extract_chat_completion_content(json.loads(body))
                return parse_teacher_json(content)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"teacher API failed with HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"teacher API request failed: {exc}") from exc

    def list_models(self) -> list[str]:
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"models probe failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"models probe request failed: {exc}") from exc
        return parse_models_response(json.loads(body))

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        if delay > 0:
            time.sleep(delay)


def build_system_prompt() -> str:
    return (
        "You generate concise solution traces for reasoning benchmark examples. "
        "Return only valid JSON with keys reasoning and final_answer. "
        "The reasoning value should be a visible solution trace that explains how the answer is derived. "
        "The final_answer value must be the shortest final answer string, without extra prose."
    )


def build_user_prompt(prompt: str, _expected_answer: str) -> str:
    return (
        "Solve the problem below. Return JSON only.\n\n"
        f"Problem:\n{prompt}\n\n"
        'JSON schema: {"reasoning": "...", "final_answer": "..."}'
    )


def extract_chat_completion_content(payload: dict[str, Any]) -> str:
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("teacher API response does not contain choices[0].message.content") from exc


def parse_models_response(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    models: list[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            models.append(str(item["id"]))
    return sorted(set(models))


def parse_teacher_json(text: str) -> dict[str, str]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("teacher response is not valid JSON") from exc
    reasoning = str(payload.get("reasoning", "")).strip()
    final_answer = str(payload.get("final_answer", "")).strip()
    if not reasoning:
        raise ValueError("teacher response is missing reasoning")
    if not final_answer:
        raise ValueError("teacher response is missing final_answer")
    return {"reasoning": reasoning, "final_answer": final_answer}


def normalize_answer(value: str) -> str:
    text = str(value).strip()
    boxed = re.search(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        text = boxed.group(1)
    text = text.strip().strip("`").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def answers_match(teacher_answer: str, expected_answer: str) -> bool:
    return normalize_answer(teacher_answer) == normalize_answer(expected_answer)


def normalize_base_url(value: str) -> str:
    url = str(value).strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a JSON object: {path}")
    return payload


def choose_config_value(
    cli_value: Any,
    config: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    if key in config and config[key] not in ("", None):
        return config[key]
    return default


def is_text_generation_model(model: str) -> bool:
    lower = model.lower()
    return not any(fragment in lower for fragment in NON_TEXT_MODEL_FRAGMENTS)


def recommend_model(models: list[str]) -> str | None:
    if not models:
        return None
    candidates = [model for model in models if is_text_generation_model(model)]
    if not candidates:
        return None
    lowered = {model: model.lower() for model in candidates}
    for preferred in PREFERRED_TEXT_MODELS:
        exact = [model for model, lower in lowered.items() if lower == preferred]
        if exact:
            return sorted(exact)[0]
        versioned = [
            model
            for model, lower in lowered.items()
            if lower.startswith(f"{preferred}-20") or lower.startswith(f"{preferred}-latest")
        ]
        if versioned:
            return sorted(versioned, reverse=True)[0]
    for fragment in ("reason", "chat", "instruct", "qwen", "deepseek", "claude"):
        matches = [model for model, lower in lowered.items() if fragment in lower]
        if matches:
            return sorted(matches, key=lambda item: (len(item), item))[0]
    return sorted(candidates)[0]


def infer_type(prompt: str) -> str:
    lower = prompt.lower()
    if "bit manipulation" in lower or "8-bit binary" in lower:
        return "bit_manipulation"
    if "secret encryption" in lower or "decrypt" in lower:
        return "encryption"
    if "transformation rules" in lower:
        return "symbol_transformation"
    return ""


def rejected_path_for(output: Path) -> Path:
    return output.with_name(f"{output.stem}.rejected{output.suffix}")


def load_existing_ids(paths: Iterable[Path]) -> set[str]:
    existing: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_id = row.get("id")
                if row_id:
                    existing.add(row_id)
    return existing


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def append_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def build_generation_record(row: dict[str, str], provider: TeacherProvider) -> tuple[dict[str, Any], bool]:
    row_id = row.get("id", "")
    prompt = row.get("prompt", "")
    answer = row.get("answer", "")
    row_type = row.get("type") or infer_type(prompt)
    try:
        result = provider.generate(prompt, answer)
        reasoning = result["reasoning"].strip()
        teacher_answer = result["final_answer"].strip()
        match = answers_match(teacher_answer, answer)
        record = {
            "id": row_id,
            "prompt": prompt,
            "answer": answer,
            "generated_cot": reasoning,
            "type": row_type,
            "teacher_answer": teacher_answer,
            "match": str(match).lower(),
            "error": "",
        }
        return record, match
    except Exception as exc:
        return (
            {
                "id": row_id,
                "prompt": prompt,
                "answer": answer,
                "generated_cot": "",
                "type": row_type,
                "teacher_answer": "",
                "match": "false",
                "error": str(exc),
            },
            False,
        )


def generate_teacher_cot(
    *,
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    provider: TeacherProvider,
    limit: int | None = None,
    resume: bool = False,
    sleep_seconds: float = 0.0,
) -> dict[str, int]:
    rows = read_rows(input_path)
    existing_ids = load_existing_ids((output_path, rejected_path)) if resume else set()
    processed = matched = rejected = skipped = 0

    for row in rows:
        row_id = row.get("id", "")
        if resume and row_id in existing_ids:
            skipped += 1
            continue
        if limit is not None and processed >= limit:
            break
        record, is_match = build_generation_record(row, provider)
        processed += 1
        if is_match:
            append_rows(output_path, MATCHED_FIELDS, [record])
            matched += 1
        else:
            append_rows(rejected_path, REJECTED_FIELDS, [record])
            rejected += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return {
        "processed": processed,
        "matched": matched,
        "rejected": rejected,
        "skipped": skipped,
    }


def probe_upstream(provider: OpenAICompatibleTeacher) -> dict[str, Any]:
    result: dict[str, Any] = {
        "base_url": provider.base_url,
        "models": [],
        "recommended_model": provider.model,
        "json_response_format": "unknown",
        "chat_completions": "unknown",
        "recommended_max_output_tokens": provider.max_output_tokens,
        "sample_command": "",
    }
    try:
        models = provider.list_models()
        result["models"] = models
        result["recommended_model"] = recommend_model(models) or provider.model
    except Exception as exc:
        result["models_error"] = str(exc)

    test_model = str(result["recommended_model"] or provider.model)
    test_provider = OpenAICompatibleTeacher(
        base_url=provider.base_url,
        model=test_model,
        api_key=provider.api_key,
        temperature=provider.temperature,
        max_output_tokens=256,
        request_timeout=DEFAULT_PROBE_TIMEOUT,
        max_retries=0,
    )
    try:
        payload = test_provider.generate(PROBE_PROMPT, PROBE_ANSWER)
        result["chat_completions"] = "ok"
        result["json_response_format"] = "ok"
        result["probe_response"] = payload
    except Exception as exc:
        result["chat_completions"] = "failed"
        result["probe_error"] = str(exc)

    result["sample_command"] = (
        "python -m src.generate_teacher_cot --limit 3 "
        f"--model {test_model} --output data/derived/problem_ids_matched.sample.csv"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate teacher-CoT traces and build problem_ids_matched.csv for LoRA SFT."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    base_url = normalize_base_url(
        choose_config_value(
            args.base_url,
            config,
            "base_url",
            os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        )
    )
    model = choose_config_value(args.model, config, "model")
    temperature = float(
        choose_config_value(args.temperature, config, "temperature", DEFAULT_TEMPERATURE)
    )
    max_output_tokens = int(
        choose_config_value(
            args.max_output_tokens,
            config,
            "max_output_tokens",
            DEFAULT_MAX_OUTPUT_TOKENS,
        )
    )
    config_limit = choose_config_value(args.limit, config, "limit")
    limit = int(config_limit) if config_limit is not None else None
    api_key = str(choose_config_value(None, config, "api_key", os.environ.get(args.api_key_env, "")))
    if not api_key:
        print(
            f"Missing API key. Fill {args.config} or set environment variable {args.api_key_env}.",
            file=sys.stderr,
        )
        return 2
    if not model and not args.probe:
        print(
            "Missing model. Run `python -m src.generate_teacher_cot --probe` first, "
            "then rerun with --model <recommended-model>.",
            file=sys.stderr,
        )
        return 2

    rejected_output = args.rejected_output or rejected_path_for(args.output)
    request_timeout = DEFAULT_PROBE_TIMEOUT if args.probe else DEFAULT_REQUEST_TIMEOUT
    max_retries = 0 if args.probe else DEFAULT_MAX_RETRIES
    provider = OpenAICompatibleTeacher(
        base_url=str(base_url),
        model=str(model or "probe-model"),
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )
    if args.probe:
        result = probe_upstream(provider)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    summary = generate_teacher_cot(
        input_path=args.input,
        output_path=args.output,
        rejected_path=rejected_output,
        provider=provider,
        limit=limit,
        resume=args.resume,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(summary, indent=2))
    print(f"matched_output={args.output}")
    print(f"rejected_output={rejected_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
