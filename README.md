# NVIDIA Nemotron Model Reasoning Challenge

**Kaggle Silver Medal · Private LB #167 · Public 0.864 / Private 0.860**

This repository presents a silver-medal solution workflow for the NVIDIA Nemotron Model Reasoning Challenge. The project centers on adapting `Nemotron-3-Nano-30B-A3B-BF16` with a rank-32 LoRA adapter for reasoning tasks, using Chain-of-Thought supervised fine-tuning and Kaggle GPU experiments.

## Result

| Metric | Value |
| --- | --- |
| Competition | NVIDIA Nemotron Model Reasoning Challenge |
| Medal | Silver |
| Private leaderboard rank | 167 |
| Public score | 0.864 |
| Private score | 0.860 |
| Team | `galaxy2025` |
| Final submission | `final_private_lb_086` |
| Submission reference | `53620005` |

![Private leaderboard row for rank 167](docs/assets/private-leaderboard-rank-167.png)

Additional leaderboard details are available in [docs/RESULTS.md](docs/RESULTS.md).

## Approach

The solution uses the official Nemotron base model as a fixed backbone and trains a LoRA adapter rather than fine-tuning all model parameters. The adapter is constrained to rank 32 and targets both attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP/expert projections (`in_proj`, `out_proj`, `up_proj`, `down_proj`), so the update can affect reasoning behavior beyond the final answer format.

Training data is converted from `prompt`, `answer`, and `generated_cot` fields into chat-style SFT examples. The user side appends a requirement that the final answer appear in `\boxed{}`. The assistant side keeps the generated reasoning trace, removes any existing boxed answer from the raw CoT, and appends a normalized `</think>` plus boxed target answer. This makes the supervision cover both intermediate reasoning and the competition answer format.

The Kaggle training experiment uses 8192-token context length, BF16, gradient checkpointing, batch size 1, and 32-step gradient accumulation. Samples are ordered with a type-stratified sampler so each effective batch is less dominated by a single problem category. The final artifact is exported as a LoRA adapter and evaluated through Kaggle submissions.

Adapter-level experiments compared single-adapter selection, SVD-based compression/reconstruction, and adapter fusion. Fusion variants underperformed substantially, with direct weight averaging dropping from the 0.86 range to about 0.51, TIES-style fusion to about 0.46, and SVD fusion to about 0.77. The final solution therefore uses a single stable adapter instead of combining low-rank updates with mismatched subspaces.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `notebooks/training_experiment/` | Kaggle GPU LoRA SFT training experiment. |
| `notebooks/final_submission/` | Kaggle script wrapper for the final submission. |
| `src/` | Utilities for adapter packaging, Kaggle submission, quota checks, and score synchronization. |
| `docs/RESULTS.md` | Final leaderboard result and screenshot. |
| `tests/` | Unit tests for packaging, submission tracking, quota calculation, and notebook generation. |

## Run Checks

```powershell
C:\Users\wangk\.conda\envs\Kaggle\python.exe -m pip install -r requirements.txt
C:\Users\wangk\.conda\envs\Kaggle\python.exe -m pytest
```

## Artifact Policy

The repository excludes Kaggle competition data, model weights, LoRA weight files, generated `submission.zip` archives, runtime logs, local caches, and credentials.

## License

This repository is released under the MIT License.
