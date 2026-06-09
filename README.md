# NVIDIA Nemotron Model Reasoning Challenge

Silver Medal | Private Leaderboard Rank 167 | Score 0.86

This repository documents a Kaggle competition workflow for the NVIDIA Nemotron Model Reasoning Challenge. The project builds, validates, submits, and tracks LoRA adapter packages for the `Nemotron-3-Nano-30B` base model under Kaggle's competition constraints.

## Result

| Field | Value |
| --- | --- |
| Competition | NVIDIA Nemotron Model Reasoning Challenge |
| Kaggle slug | `nvidia-nemotron-model-reasoning-challenge` |
| Medal | Silver |
| Private leaderboard rank | 167 |
| Private leaderboard score | 0.86 |
| Team | `galaxy2025` |
| Submissions | 34 |
| Final selected candidate | `mirza_lora_safetensors_output` |
| Selected candidate scores | Public `0.86`, private `0.86` |

Private leaderboard evidence:

![Private leaderboard rank 167](docs/assets/private-leaderboard-rank-167.png)

## Approach

The competition submission format is a LoRA adapter archive named `submission.zip`. The archive must contain root-level adapter files for `Nemotron-3-Nano-30B`, including `adapter_config.json` and `adapter_model.safetensors`, with LoRA rank not exceeding `32`.

The project workflow focused on operational reliability and private leaderboard robustness:

- Adapter package validation before every submission.
- Kaggle API and CLI automation for download, packaging, submission, polling, quota checks, and score synchronization.
- A candidate registry with source, score, status, rank, submission ref, and generalization notes.
- Daily submission batches that avoided relying only on one public-notebook lineage.
- Negative-result tracking for failed repacks, invalid adapter layouts, and unsuccessful adapter-space merges.

The final private leaderboard score came from a validated public adapter artifact in the Mirza adapter family. Earlier batches compared Huikang/SVD, Kienngx/Tinker, Finding Nemo, Keith bridge anchors, direct public model adapters, TIES merge output, and SVD fusion variants.

## Repository Contents

| Path | Contents |
| --- | --- |
| `candidate_registry.json` | Candidate ledger with submission refs, public/private scores, status, and selection notes. |
| `STRATEGY.md` | Post-competition strategy report and batch timeline. |
| `src/config.py` | Competition slug, artifact paths, and source constants. |
| `src/download_data.py` | Official competition data download and extraction. |
| `src/inspect_data.py` | CSV shape and field inspection. |
| `src/download_public_adapter.py` | Kaggle Model adapter download. |
| `src/package_adapter.py` | Adapter validation and `submission.zip` creation. |
| `src/kaggle_submit.py` | Submission, polling, and local JSONL logging. |
| `src/check_submissions.py` | Submission listing and best-score reporting. |
| `src/sync_candidate_registry.py` | Candidate registry synchronization from Kaggle submission rows. |
| `src/submission_quota.py` | UTC daily submission quota status from Kaggle metadata. |
| `src/create_remote_submit_notebook.py` | Kaggle script-notebook generator for remote output repacking and submission. |
| `src/notebook_templates.py` | Remote Kaggle submission wrapper template. |
| `notebooks/` | Generated Kaggle script-kernel wrappers used for candidate submissions. |
| `tests/` | Unit tests for packaging, submission matching, quota calculation, notebook generation, and registry sync. |

## Setup

The local workflow uses the `Kaggle` Conda environment:

```powershell
C:\Users\wangk\.conda\envs\Kaggle\python.exe -m pip install -r requirements.txt
```

The Kaggle CLI executable is expected at:

```powershell
C:\Users\wangk\.conda\envs\Kaggle\Scripts\kaggle.exe
```

`KAGGLE_USERNAME` and `KAGGLE_KEY` can be provided through the normal Kaggle API configuration. Credentials are not stored in this repository.

## Workflow

Download and inspect competition data:

```powershell
python -m src.download_data
python -m src.inspect_data
```

Download and package the baseline public adapter:

```powershell
python -m src.download_public_adapter
python -m src.package_adapter
```

Submit a package and record the result:

```powershell
python -m src.kaggle_submit --message "public adapter v1 9500s-batch1-lr1e-4 quick baseline"
```

Check recent submissions and synchronize the candidate registry:

```powershell
python -m src.check_submissions --page-size 100
python -m src.sync_candidate_registry --page-size 100
```

Generate a Kaggle remote-submit wrapper:

```powershell
python -m src.create_remote_submit_notebook `
  --slug submit_example_candidate `
  --kernel-source author/source-kernel `
  --message "public example candidate output"
```

Check the daily UTC quota:

```powershell
python -m src.submission_quota
```

Run tests:

```powershell
python -m pytest
```

## Artifact Policy

Generated artifacts are excluded from version control:

- Official competition data under `data/raw/`.
- Downloaded models, adapter weights, and `.safetensors` files.
- `submission.zip` and other generated archives.
- Submission logs and Kaggle runtime logs.
- Notebook outputs, Kaggle caches, temporary folders, and local environment files.

The repository stores code, metadata, strategy notes, candidate records, and a cropped private-leaderboard evidence image. It does not redistribute competition data or model weights.

## License

This repository is released under the MIT License. See `LICENSE` for details.
