from __future__ import annotations

import math

import pytest

from notebooks.training_experiment.kaggle_training_experiment import (
    IGNORE_INDEX,
    WeightedDataCollator,
    assert_tokenizer_supports_offsets,
    prepare_sft_example,
    strip_boxed_answers,
    weighted_causal_lm_loss,
)


class CharOffsetTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.last_text = ""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert not tokenize
        assert not add_generation_prompt
        user = messages[0]["content"]
        assistant = messages[1]["content"]
        return f"<user>{user}</user><assistant>{assistant}</assistant>"

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, truncation=False):
        assert not add_special_tokens
        assert not truncation
        self.last_text = text
        encoded = {
            "input_ids": [ord(char) for char in text],
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return encoded


class NoOffsetTokenizer(CharOffsetTokenizer):
    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, truncation=False):
        encoded = super().__call__(
            text,
            return_offsets_mapping=False,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
        )
        return encoded


def supervised_text(example) -> str:
    return "".join(chr(label) for label in example.labels if label != IGNORE_INDEX)


def test_response_only_mask_and_answer_suffix_weight_use_offsets() -> None:
    tokenizer = CharOffsetTokenizer()

    example = prepare_sft_example(
        tokenizer=tokenizer,
        prompt="What is 2+2?",
        answer="4",
        cot_text="Compute 2+2.",
        sample_id="row-1",
        problem_type="math",
        max_length=4096,
        answer_weight=2.0,
    )

    text = tokenizer.last_text
    assistant_start = text.index("Compute 2+2.")
    answer_start = text.index("</think>\n\\boxed{4}")

    assert all(label == IGNORE_INDEX for label in example.labels[:assistant_start])
    assert supervised_text(example) == "Compute 2+2.\n</think>\n\\boxed{4}"
    assert all(weight == 1.0 for weight in example.loss_weights[assistant_start:answer_start])
    assert all(weight == 2.0 for weight in example.loss_weights[answer_start : answer_start + len("</think>\n\\boxed{4}")])


def test_answer_preserving_truncation_uses_token_budget() -> None:
    tokenizer = CharOffsetTokenizer()
    empty = prepare_sft_example(
        tokenizer=tokenizer,
        prompt="P",
        answer="42",
        cot_text="",
        sample_id="empty",
        problem_type="math",
        max_length=4096,
    )
    max_length = len(empty.input_ids) + 8

    example = prepare_sft_example(
        tokenizer=tokenizer,
        prompt="P",
        answer="42",
        cot_text="a" * 200,
        sample_id="long",
        problem_type="math",
        max_length=max_length,
    )

    assert example.truncated
    assert len(example.input_ids) <= max_length
    assert "</think>\n\\boxed{42}" in supervised_text(example)
    assert example.answer_weighted_tokens == len("</think>\n\\boxed{42}")
    assert supervised_text(example).startswith("a")


def test_prompt_plus_answer_over_budget_fails_fast() -> None:
    tokenizer = CharOffsetTokenizer()

    with pytest.raises(ValueError, match="exceeding max_length"):
        prepare_sft_example(
            tokenizer=tokenizer,
            prompt="P" * 200,
            answer="42",
            cot_text="short cot",
            sample_id="too-long",
            problem_type="math",
            max_length=10,
        )


def test_tokenizer_offset_support_check_fails_without_offsets() -> None:
    assert_tokenizer_supports_offsets(CharOffsetTokenizer())

    with pytest.raises(RuntimeError, match="offset_mapping"):
        assert_tokenizer_supports_offsets(NoOffsetTokenizer())


def test_strip_boxed_answers_removes_teacher_boxed_suffix() -> None:
    assert strip_boxed_answers("Reasoning. \\boxed{123}  ") == "Reasoning."


def test_weighted_causal_lm_loss_shifts_labels_and_weights() -> None:
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")

    logits = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[IGNORE_INDEX, 0, IGNORE_INDEX, 2]], dtype=torch.long)
    loss_weights = torch.tensor([[99.0, 1.0, 50.0, 4.0]], dtype=torch.float32)

    loss = weighted_causal_lm_loss(logits, labels, loss_weights)
    shifted_logits = logits[:, :-1, :].contiguous()
    expected_token_loss = functional.cross_entropy(
        shifted_logits.view(-1, 3),
        torch.tensor([0, 0, 2]),
        reduction="none",
    )
    expected = (expected_token_loss * torch.tensor([1.0, 0.0, 4.0])).sum() / 5.0

    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-6)


def test_weighted_data_collator_pads_labels_and_weights() -> None:
    torch = pytest.importorskip("torch")
    collator = WeightedDataCollator(pad_token_id=0)

    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [IGNORE_INDEX, 2], "loss_weights": [0.0, 1.0]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3], "loss_weights": [2.0]},
        ]
    )

    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [3, 0]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[IGNORE_INDEX, 2], [3, IGNORE_INDEX]]))
    assert torch.equal(batch["loss_weights"], torch.tensor([[0.0, 1.0], [2.0, 0.0]]))
