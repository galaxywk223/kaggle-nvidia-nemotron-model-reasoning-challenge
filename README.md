# NVIDIA Nemotron Model Reasoning Challenge

**Silver Medal · Private LB #167 · Score 0.86**

Silver-medal Kaggle solution toolkit for LoRA adapter validation, submission automation, and candidate tracking in the NVIDIA Nemotron Model Reasoning Challenge.

## Result

| Metric | Value |
| --- | --- |
| Competition | NVIDIA Nemotron Model Reasoning Challenge |
| Medal | Silver |
| Private leaderboard rank | 167 |
| Private leaderboard score | 0.86 |
| Team | `galaxy2025` |
| Submissions | 34 |
| Final candidate | `mirza_lora_safetensors_output` |

![Private leaderboard row for rank 167](docs/assets/private-leaderboard-rank-167.png)

Additional result evidence is available in [docs/RESULTS.md](docs/RESULTS.md).

## What This Repository Contains

This repository packages the competition work as a reusable submission toolkit for adapter-based Kaggle competitions:

- LoRA adapter validation for required files, archive layout, and rank limits.
- Kaggle submission automation with polling, status capture, and score recording.
- Remote Kaggle script generation for repacking public kernel or model artifacts.
- Candidate tracking with public/private scores and submission references.
- Tests covering packaging, submission matching, quota calculation, registry synchronization, and notebook generation.

## Competition Constraints

| Constraint | Value |
| --- | --- |
| Submission artifact | `submission.zip` |
| Adapter format | LoRA adapter for `Nemotron-3-Nano-30B` |
| Required root files | `adapter_config.json`, `adapter_model.safetensors` |
| Maximum LoRA rank | 32 |
| Daily submission limit | 5 |

## Project Layout

| Path | Purpose |
| --- | --- |
| `src/` | Python toolkit for data inspection, adapter packaging, submission, quota checks, and registry sync. |
| `notebooks/` | Kaggle script-kernel wrappers used to submit or repack candidate artifacts. |
| `candidate_registry.json` | Machine-readable candidate ledger with Kaggle refs and scores. |
| `docs/RESULTS.md` | Final private leaderboard evidence. |
| `tests/` | Unit tests for the reusable tooling. |

## Reproducibility

The repository is intended to reproduce the tooling path, not to redistribute Kaggle data or model weights.

```powershell
C:\Users\wangk\.conda\envs\Kaggle\python.exe -m pip install -r requirements.txt
C:\Users\wangk\.conda\envs\Kaggle\python.exe -m pytest
```

Core entry points:

```powershell
python -m src.package_adapter
python -m src.kaggle_submit --message "submission message"
python -m src.check_submissions --page-size 100
python -m src.sync_candidate_registry --page-size 100
python -m src.create_remote_submit_notebook --slug submit_example --kernel-source author/kernel --message "candidate message"
```

Kaggle credentials are expected through the standard Kaggle API configuration.

## Artifact Policy

The repository excludes generated or restricted artifacts:

- Official competition data.
- Downloaded models and adapter weights.
- `submission.zip` archives.
- Runtime logs and local Kaggle caches.
- Local environment files and credentials.

## License

This repository is released under the MIT License.
