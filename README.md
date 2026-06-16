# NVIDIA Nemotron Model Reasoning Challenge

**Silver Medal · Private LB #167 · Score 0.86**

Silver-medal Kaggle solution toolkit for a final private leaderboard submission in the NVIDIA Nemotron Model Reasoning Challenge.

## Result

| Metric | Value |
| --- | --- |
| Competition | NVIDIA Nemotron Model Reasoning Challenge |
| Medal | Silver |
| Private leaderboard rank | 167 |
| Private leaderboard score | 0.86 |
| Team | `galaxy2025` |
| Final submission | `final_private_lb_086` |
| Final candidate scores | Public `0.86`, private `0.86` |

![Private leaderboard row for rank 167](docs/assets/private-leaderboard-rank-167.png)

Additional result evidence is available in [docs/RESULTS.md](docs/RESULTS.md).

## What This Repository Contains

This repository preserves the best-scoring submission path and the reusable tooling around it:

- LoRA adapter validation for required files, archive layout, and rank limits.
- Kaggle submission automation with polling, status capture, and score recording.
- A remote Kaggle script wrapper for the final best-scoring submission.
- A minimal candidate record for the final public/private `0.86` submission.
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
| `src/` | Python toolkit for adapter packaging, Kaggle submission, quota checks, and registry sync. |
| `notebooks/final_submission/` | Final Kaggle submission wrapper. |
| `candidate_registry.json` | Single-record ledger for the final public/private `0.86` candidate. |
| `docs/RESULTS.md` | Private leaderboard evidence. |
| `tests/` | Unit tests for the reusable tooling. |

## Reproducibility

The repository is intended to reproduce the tooling path for the retained best submission. It does not redistribute Kaggle data or model weights.

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
