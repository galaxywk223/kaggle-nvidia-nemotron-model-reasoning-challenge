from __future__ import annotations

import csv
import json
import urllib.error
from pathlib import Path

from src.generate_teacher_cot import (
    answers_match,
    build_user_prompt,
    choose_config_value,
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    generate_teacher_cot,
    load_config,
    normalize_base_url,
    normalize_answer,
    parse_teacher_json,
    parse_models_response,
    probe_upstream,
    recommend_model,
    rejected_path_for,
)


class FakeTeacher:
    def __init__(self, responses: dict[str, dict[str, str]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def generate(self, prompt: str, expected_answer: str) -> dict[str, str]:
        self.calls.append(prompt)
        if prompt not in self.responses:
            raise RuntimeError("missing fake response")
        return self.responses[prompt]


class FakeProbeTeacher:
    def __init__(self, models: list[str], response: dict[str, str]) -> None:
        self.base_url = "https://example.test/v1"
        self.model = "probe-model"
        self.api_key = "test-key"
        self.temperature = 0.2
        self.max_output_tokens = 2048
        self.request_timeout = 120
        self.max_retries = 0
        self.retry_backoff_seconds = 0.0
        self.models = models
        self.response = response

    def list_models(self) -> list[str]:
        return self.models

    def generate(self, prompt: str, expected_answer: str) -> dict[str, str]:
        return self.response


def write_train_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_parse_teacher_json_accepts_plain_and_fenced_json() -> None:
    payload = parse_teacher_json('{"reasoning": "trace", "final_answer": "42"}')
    fenced = parse_teacher_json('```json\n{"reasoning": "trace", "final_answer": "42"}\n```')

    assert payload == {"reasoning": "trace", "final_answer": "42"}
    assert fenced == payload


def test_normalize_answer_handles_boxed_quotes_spaces_and_case() -> None:
    assert normalize_answer(r" \boxed{ Cat Imagines Book } ") == "cat imagines book"
    assert normalize_answer("`10010111`") == "10010111"
    assert normalize_answer("cat\nimagines\tbook") == "cat imagines book"


def test_answers_match_after_normalization() -> None:
    assert answers_match(r"\boxed{10010111}", "10010111")
    assert answers_match("Cat Imagines Book", "cat imagines book")
    assert not answers_match("10010110", "10010111")


def test_generate_teacher_cot_writes_matched_and_rejected_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "problem_ids_matched.csv"
    rejected_path = rejected_path_for(output_path)
    write_train_csv(
        input_path,
        [
            {"id": "a", "prompt": "bit manipulation prompt", "answer": "10010111"},
            {"id": "b", "prompt": "decrypt prompt", "answer": "cat imagines book"},
        ],
    )
    provider = FakeTeacher(
        {
            "bit manipulation prompt": {
                "reasoning": "Infer the bit rule and apply it.",
                "final_answer": r"\boxed{10010111}",
            },
            "decrypt prompt": {
                "reasoning": "Infer the substitution and decode the phrase.",
                "final_answer": "cat imagines books",
            },
        }
    )

    summary = generate_teacher_cot(
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path,
        provider=provider,
    )

    assert summary == {"processed": 2, "matched": 1, "rejected": 1, "skipped": 0}
    matched = read_csv(output_path)
    rejected = read_csv(rejected_path)
    assert matched[0]["id"] == "a"
    assert matched[0]["generated_cot"] == "Infer the bit rule and apply it."
    assert matched[0]["type"] == "bit_manipulation"
    assert "teacher_answer" not in matched[0]
    assert rejected[0]["id"] == "b"
    assert rejected[0]["teacher_answer"] == "cat imagines books"
    assert rejected[0]["match"] == "false"


def test_generate_teacher_cot_resume_skips_existing_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "problem_ids_matched.csv"
    rejected_path = rejected_path_for(output_path)
    write_train_csv(
        input_path,
        [
            {"id": "a", "prompt": "first prompt", "answer": "A"},
            {"id": "b", "prompt": "second prompt", "answer": "B"},
        ],
    )
    output_path.write_text(
        "id,prompt,answer,generated_cot,type\n"
        "a,first prompt,A,existing trace,\n",
        encoding="utf-8",
    )
    provider = FakeTeacher(
        {
            "second prompt": {
                "reasoning": "second trace",
                "final_answer": "B",
            }
        }
    )

    summary = generate_teacher_cot(
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path,
        provider=provider,
        resume=True,
    )

    assert summary == {"processed": 1, "matched": 1, "rejected": 0, "skipped": 1}
    assert provider.calls == ["second prompt"]
    assert [row["id"] for row in read_csv(output_path)] == ["a", "b"]


def test_generate_teacher_cot_respects_limit(tmp_path: Path) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "problem_ids_matched.csv"
    write_train_csv(
        input_path,
        [
            {"id": "a", "prompt": "first prompt", "answer": "A"},
            {"id": "b", "prompt": "second prompt", "answer": "B"},
        ],
    )
    provider = FakeTeacher(
        {
            "first prompt": {
                "reasoning": "first trace",
                "final_answer": "A",
            },
            "second prompt": {
                "reasoning": "second trace",
                "final_answer": "B",
            },
        }
    )

    summary = generate_teacher_cot(
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path_for(output_path),
        provider=provider,
        limit=1,
    )

    assert summary["processed"] == 1
    assert provider.calls == ["first prompt"]
    assert len(read_csv(output_path)) == 1


def test_teacher_json_payload_matches_expected_keys() -> None:
    text = json.dumps({"reasoning": "Use the examples to infer a rule.", "final_answer": "@&"})

    assert parse_teacher_json(text)["final_answer"] == "@&"


def test_build_user_prompt_does_not_leak_ground_truth_answer() -> None:
    prompt = build_user_prompt("Solve this hidden rule.", "SECRET_ANSWER")

    assert "Solve this hidden rule." in prompt
    assert "SECRET_ANSWER" not in prompt
    assert "ground-truth" not in prompt.lower()


def test_load_config_and_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "teacher_cot.local.json"
    config_path.write_text(
        json.dumps({"base_url": "https://provider.test/v1", "api_key": "secret"}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["api_key"] == "secret"
    assert choose_config_value(None, config, "base_url") == "https://provider.test/v1"
    assert choose_config_value("https://override.test/v1", config, "base_url") == "https://override.test/v1"


def test_parse_models_response_and_recommend_model() -> None:
    models = parse_models_response(
        {
            "data": [
                {"id": "small-chat"},
                {"id": "gpt-5.5"},
                {"id": "gpt-5.4-mini"},
                {"id": "gpt-4o-audio-preview"},
                {"id": "embedding-model"},
            ]
        }
    )

    assert models == ["embedding-model", "gpt-4o-audio-preview", "gpt-5.4-mini", "gpt-5.5", "small-chat"]
    assert recommend_model(models) == "gpt-5.5"


def test_normalize_base_url_adds_v1_once() -> None:
    assert normalize_base_url("https://provider.test") == "https://provider.test/v1"
    assert normalize_base_url("https://provider.test/") == "https://provider.test/v1"
    assert normalize_base_url("https://provider.test/v1") == "https://provider.test/v1"


def test_recommend_model_filters_non_text_models() -> None:
    assert recommend_model(["gpt-image-1", "gpt-4o-audio-preview", "codex-auto-review"]) is None
    assert recommend_model(["gpt-5.4-mini", "gpt-4o-audio-preview"]) == "gpt-5.4-mini"


def test_probe_upstream_reports_recommendation(monkeypatch) -> None:
    fake = FakeProbeTeacher(
        models=["gpt-4o-audio-preview", "gpt-5.4", "gpt-5.5"],
        response={"reasoning": "2 + 3 = 5.", "final_answer": "5"},
    )

    def build_provider(**kwargs):
        assert kwargs["model"] == "gpt-5.5"
        return fake

    monkeypatch.setattr("src.generate_teacher_cot.OpenAICompatibleTeacher", build_provider)

    result = probe_upstream(fake)  # type: ignore[arg-type]

    assert result["recommended_model"] == "gpt-5.5"
    assert result["chat_completions"] == "ok"
    assert result["json_response_format"] == "ok"
    assert "--model gpt-5.5" in result["sample_command"]


def test_probe_upstream_uses_short_timeout_and_single_attempt(monkeypatch) -> None:
    class RecordingProbeTeacher(FakeProbeTeacher):
        def __init__(self) -> None:
            super().__init__(
                models=["gpt-5.5"],
                response={"reasoning": "2 + 3 = 5.", "final_answer": "5"},
            )

    fake = RecordingProbeTeacher()
    seen: dict[str, object] = {}

    def build_provider(**kwargs):
        seen.update(kwargs)
        return fake

    monkeypatch.setattr("src.generate_teacher_cot.OpenAICompatibleTeacher", build_provider)

    result = probe_upstream(fake)  # type: ignore[arg-type]

    assert result["chat_completions"] == "ok"
    assert seen["request_timeout"] == DEFAULT_PROBE_TIMEOUT
    assert seen["max_retries"] == 0


def test_probe_upstream_handles_missing_models_endpoint(monkeypatch) -> None:
    class NoModelsTeacher(FakeProbeTeacher):
        def list_models(self) -> list[str]:
            raise RuntimeError("models endpoint unavailable")

    fake = NoModelsTeacher(
        models=[],
        response={"reasoning": "2 + 3 = 5.", "final_answer": "5"},
    )

    def build_provider(**kwargs):
        return fake

    monkeypatch.setattr("src.generate_teacher_cot.OpenAICompatibleTeacher", build_provider)

    result = probe_upstream(fake)  # type: ignore[arg-type]

    assert "models_error" in result
    assert result["chat_completions"] == "ok"


def test_generate_retries_retryable_http_errors(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "problem_ids_matched.csv"
    write_train_csv(
        input_path,
        [{"id": "a", "prompt": "first prompt", "answer": "A"}],
    )

    attempts = {"count": 0, "sleeps": []}

    class FakeResponse:
        def __init__(self, body: str) -> None:
            self.body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self.body.encode("utf-8")

    provider = generate_teacher_cot.__globals__["OpenAICompatibleTeacher"](
        base_url="https://provider.test/v1",
        model="gpt-5.5",
        api_key="secret",
        temperature=0.2,
        max_output_tokens=256,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )

    def fake_urlopen(request, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.URLError("temporary network error")
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"reasoning": "first trace", "final_answer": "A"}
                            )
                        }
                    }
                ]
            }
        )
        return FakeResponse(body)

    def fake_sleep(seconds: float) -> None:
        attempts["sleeps"].append(seconds)

    monkeypatch.setattr("src.generate_teacher_cot.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.generate_teacher_cot.time.sleep", fake_sleep)

    summary = generate_teacher_cot(
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path_for(output_path),
        provider=provider,
    )

    assert summary == {"processed": 1, "matched": 1, "rejected": 0, "skipped": 0}
    assert attempts["count"] == 2
    assert attempts["sleeps"] == []
