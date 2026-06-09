# Submission Strategy

## Objective

The project targets a strong private leaderboard score while controlling public leaderboard selection risk. The workflow uses public Kaggle artifacts as references, validates each adapter package before submission, and keeps the submitted candidate set diversified across independent adapter sources.

## Final Result

| Field | Value |
| --- | --- |
| Medal | Silver |
| Private leaderboard rank | `167` |
| Private leaderboard score | `0.86` |
| Team | `galaxy2025` |
| Submission count | `34` |
| Final selected candidate | `mirza_lora_safetensors_output` |
| Final selected scores | Public `0.86`, private `0.86` |

The private leaderboard result is the primary reported competition outcome. Public leaderboard scores remain useful for triage and candidate comparison, but the final repository headline uses the private leaderboard rank and score.

## Selection Criteria

Candidate selection follows these constraints:

- The artifact must be a valid Nemotron-3-Nano-30B LoRA adapter package with `adapter_config.json` and `adapter_model.safetensors` at the `submission.zip` root.
- The adapter rank must not exceed the competition maximum rank of `32`.
- Public leaderboard evidence is considered useful but not sufficient by itself.
- The selected daily batch should include independent sources instead of repeated copies from one notebook lineage.
- Candidate notes must record public leaderboard selection risk and private leaderboard generalization considerations.
- Direct public-test hand labeling or row-level public-test replay logic is not used as a selection principle.

## Submitted Batches

### 2026-06-08 UTC

| Candidate | Source | Ref | Status | Public score | Role |
| --- | --- | --- | --- | --- | --- |
| `public_adapter_v1_9500s` | `kienngx/nemotron-nano-30b-trained/Transformers/9500s-batch1-lr1e-4/1` | `53478358` | COMPLETE | `0.53` | Workflow baseline |
| `agi_087_svd_kernel_output` | `hammadfarooq470/agi-for-medal-0-87` | `53483517` | COMPLETE | `0.85` | High-score SVD adapter line |
| `mirza_086_kernel_output` | `mirzayasirabdullah07/best-nvidia-nemotron-0-86` | `53483623` | COMPLETE | `0.86` | Independent adapter line |
| `mohamed_replay_086_kernel_output` | `mohamedamr992/nemotron-replay-data-0-86` | `53483673` | COMPLETE | `0.86` | Best completed candidate |
| `taha_086_kernel_output` | `tahaalam2009/end-to-end-finetuning-for-lb-0-86-custom-repo` | `53484402` | COMPLETE | `0.86` | Fifth independent candidate |

### 2026-06-09 UTC

| Candidate | Source | Ref | Status | Public score | Role |
| --- | --- | --- | --- | --- | --- |
| `biohack_v62_sparse_trust_output` | `biohack44/nemotron-v62-d3-sparse-trust-finisher-attack` | `53491105` | COMPLETE | `0.85` | Recent sparse-trust candidate |
| `kuang_087_training_output` | `kuangyicheng/nemotron-087-training` | `53491110` | COMPLETE | `0.84` | Public 0.87 training output |
| `johnjanson_087_output` | `johnjanson/agi-for-medal-0-87-is-possible` | `53491111` | COMPLETE | `0.85` | High-vote 0.87 public output |
| `afr1ste_086_tinker_output` | `afr1ste/nemotron-0-86-tinker-adapter-guide` | `53491171` | COMPLETE | `0.86` | Tinker-line comparison |
| `custom_diverse_086_tensor_average` | `mirza+mohamed+taha weighted adapter tensor average` | `53491333` | COMPLETE | `0.51` | Custom source-diverse blend |

### 2026-06-10 UTC

| Candidate | Source | Ref | Status | Public score | Role |
| --- | --- | --- | --- | --- | --- |
| `matthew_steinifrank_output` | `matthewblakeward/steinifrank` | `53520850` | COMPLETE | `0.51` | Independent SteinIFrank candidate |
| `bbob_tier1_lora_baseline_output` | `bbobwayne/nemotron-tier-1-lora-baseline` | `53520881` | COMPLETE | `0.85` | Conservative Tier-1 comparison |
| `bbob_tier2_unsloth_r32_output` | `bbobwayne/nemotron-tier-2-unsloth-lora-r-32` | `53520890` | COMPLETE | `0.83` | Stronger Tier-2 comparison |
| `hammad_structured_reasoning_sft_output` | `hammadfarooq470/structured-reasoning-via-sft` | `53520926` | COMPLETE | `0.50` | Structured reasoning SFT candidate |
| `banwait_steroids_output` | `banwait13/nemotron-on-steroids` | `53520971` | COMPLETE | `0.49` | Independent aggressive public output |

### 2026-06-11 UTC

| Candidate | Source | Ref | Status | Public score | Role |
| --- | --- | --- | --- | --- | --- |
| `evgendvorkin_adapter_output` | `evgendvorkin/nemotron-3-nano-lora-adapter-submission` | `53560958` | COMPLETE | `0.64` | Recent lightweight independent adapter output |
| `svanik_engine_ensembler_repacked_output` | `svanikkolli/nemotron-engine-ensembler` | `53560987` | ERROR | - | Engine-ensembler source-diversity candidate |
| `vng_refine_svd_output` | `vngnguynhuy/refine` | `53561098` | COMPLETE | `0.85` | Recent Huikang/Tinker SVD refine implementation |
| `rauff_modular_v2_output` | `rauffauzanrambe/lora-nemo-pipeline-improve-modular-v2` | `53561151` | COMPLETE | `0.58` | Recent modular LoRA compression pipeline |
| `paritosh_svd_denoise_output` | `paritoshtripathi5/nvidia-nemotron-svd` | `53561152` | COMPLETE | `0.64` | SVD denoising transformation candidate |

### 2026-06-13 UTC

| Candidate | Source | Ref | Status | Public score | Role |
| --- | --- | --- | --- | --- | --- |
| `mirza_top_scorer_20260612_output` | `mirzayasirabdullah07/top-scorer-nvidia-nemotron-competition` | `53620005` | COMPLETE | `0.86` | Recent Mirza output artifact with public top-scorer claim |
| `keith_hk_default20_ready_anchor` | `keithtyser/nemotron-086-adapters-20260605/Transformers/huikang_default20_ready/1` | `53620122` | COMPLETE | `0.85` | Conservative Huikang default20 ready anchor |
| `ayomide_finding_nemo_output` | `ayomide2000/finding-nemo` | `53620151` | COMPLETE | `0.86` | Finding Nemo adapter conversion output |
| `keith_hk_to_kn_lm_head_anchor` | `keithtyser/nemotron-086-adapters-20260605/Transformers/public_hk_to_kn_lm_head_lam1p0/1` | `53620249` | COMPLETE | `0.84` | Structured Huikang-to-Kienngx lm_head bridge |
| `keith_kn_to_hk_lm_head_alpha32_anchor` | `keithtyser/nemotron-086-adapters-20260605/Transformers/public_kn_to_hk_lm_head_lam1p0_alpha32/1` | `53620267` | COMPLETE | `0.85` | Reciprocal Kienngx-to-Huikang lm_head alpha32 bridge |

### 2026-06-14 UTC

| Candidate | Source | Ref | Status | Public score | Role |
| --- | --- | --- | --- | --- | --- |
| `keith_hk_to_kn_no_experts_anchor` | `keithtyser/nemotron-086-adapters-20260605/Transformers/public_hk_to_kn_no_experts_lam1p0/1` | `53657284` | COMPLETE | `0.86` | No-experts bridge ablation against full public-anchor variants |
| `keith_hk_to_kn_mamba_anchor` | `keithtyser/nemotron-086-adapters-20260605/Transformers/public_hk_to_kn_mamba_lam1p0/1` | `53657294` | COMPLETE | `0.85` | Mamba-focused Huikang-to-Kienngx bridge anchor |
| `ramkan_lora_adaptor_model` | `ramkan07/nemotron-lora-adaptor/PyTorch/default/1` | `53657339` | COMPLETE | `0.85` | Independent direct public model adapter |
| `keith_kn_to_hk_lm_head_anchor` | `keithtyser/nemotron-086-adapters-20260605/Transformers/public_kn_to_hk_lm_head_lam1p0/1` | `53657363` | COMPLETE | `0.85` | Non-alpha reciprocal lm_head bridge control |
| `quincy_public_models_adapter` | `quincyqiang/public-models-nemotron/Transformers/default/1` | `53657603` | COMPLETE | `0.78` | Independent direct public model adapter replacement for failed Kaichengyu wrapper |

### 2026-06-15 UTC

| Candidate | Source | Ref | Status | Public score | Private score | Role |
| --- | --- | --- | --- | --- | --- | --- |
| `dedquoc_svd_fusion_output` | `dedquoc/nvidia-nmrc-low-rank-svd-lora-adapter-fusion` | `53699330` | COMPLETE | `0.77` | `0.78` | QR-SVD LoRA fusion retry after prior quota exhaustion |
| `lopure_ties_ensemble_output` | `lopure/lora-adapter-ensembling-experiments` | `53699342` | COMPLETE | `0.46` | `0.48` | TIES merge test after naive tensor averaging failed |
| `habanwer_atlas_output` | `habanwer/nemotron-atlas` | `53699362` | COMPLETE | `0.63` | `0.58` | Independent solver-augmented SFT output |
| `cocoaai_huikang_087_svd_output` | `cocoaai/nvidia-nemotron-huikang-0-87-svd-submit` | `53699376` | COMPLETE | `0.85` | `0.85` | AGI/Huikang SVD high-score-line control |
| `mirza_lora_safetensors_output` | `mirzayasirabdullah07/nvidia-nemotron-lora-adapter-model-safetensors` | `53699380` | COMPLETE | `0.86` | `0.86` | Final private leaderboard candidate |

### 2026-06-11 UTC Non-Submitted Attempts

| Candidate | Source | Status | Reason |
| --- | --- | --- | --- |
| `debat_086_repacked_output` | `debatreyabiswas/nemotroncomp-best0-86-solution-nvidia-under-5min` | KERNEL_ERROR_NO_VALID_ADAPTER_OUTPUT | Public output zip lacked root-level adapter files and no visible adapter directory was available for repacking. |
| `safar_086_repacked_output` | `safar1/lb-score-0-86` | KERNEL_ERROR_NO_VALID_ADAPTER_OUTPUT | Public output zip lacked root-level adapter files and no visible adapter directory was available for repacking. |
| `dedquoc_svd_fusion_output` | `dedquoc/nvidia-nmrc-low-rank-svd-lora-adapter-fusion` | RETRIED_2026_06_15 | Kernel remained running after the fifth 2026-06-11 UTC submission reached the Kaggle table, then was retried on 2026-06-15 UTC as ref `53699330`. |
| `koushik_verify_finding_nemo_output` | `koushikrudra/nemotron-verify-finding-nemo` | KERNEL_RUNNING_AFTER_QUOTA_EXHAUSTION | Verification wrapper was pushed after quota pressure but did not create a competition submission before the 2026-06-13 UTC table reached five rows. |

## Overfitting Controls

The daily batch avoids choosing only the single highest public notebook title. The selected set covers several source lineages and keeps exact Kaggle submission refs in `candidate_registry.json` for final public/private score comparison. The best public score is treated as a candidate-selection signal, not proof of private leaderboard quality.

The 2026-06-09 UTC batch combines high-public-score candidates, recent independent public outputs, a Tinker-line comparison candidate, and a custom weighted tensor average across three 0.86 adapters. The custom blend is a smoothing experiment intended to test whether source-diverse LoRA factors can preserve shared useful signal while reducing reliance on one public-LB-tuned adapter.

The custom tensor average scored `0.51`, so direct averaging of LoRA factors is not retained as a promising direction. The 2026-06-10 UTC batch shifted back to validated public adapter packages from distinct authors and training lines, with paired Tier-1/Tier-2 candidates used as a stability check. The batch did not improve the public best score: `bbob_tier1_lora_baseline_output` reached `0.85`, `bbob_tier2_unsloth_r32_output` reached `0.83`, and the remaining candidates scored between `0.49` and `0.51`.

The 2026-06-11 UTC batch tests repacked public output artifacts and recent notebook outputs while excluding sources with explicit public-test replay signals. The `wethepeople918/nemotronloraforge` notebook was not selected because its source includes an exact train/test replay memory index, which is incompatible with private leaderboard robustness. The remote submit template validates root-level `submission.zip` files first and falls back to repacking a discovered `adapter_config.json` plus `adapter_model.safetensors` directory when the public output zip has nested paths. `debat_086_repacked_output` and `safar_086_repacked_output` were rejected by this validation because their public outputs did not expose a valid adapter package for the wrapper. The completed 2026-06-11 candidates did not improve the `0.86` public best; `vng_refine_svd_output` reached `0.85`, the lower-scoring repack/compression variants reached `0.58` to `0.64`, and `svanik_engine_ensembler_repacked_output` errored.

The 2026-06-13 UTC batch keeps the direct averaging rollback in force and tests packaged public artifacts or structured 0.86 adapter anchors instead. The Mirza top-scorer artifact is useful as a recent packaging check but is not independent from the existing Mirza 0.86 line. The Keith anchor variants provide conservative and reciprocal bridge checks across the Huikang/Kienngx lines, with public-adapter-derived selection bias recorded in `candidate_registry.json`. The Finding Nemo output provides a conversion/compression check without explicit public-test replay logic in the reviewed source. The completed batch did not improve the public best: Mirza and Finding Nemo matched `0.86`, while the Keith anchors scored `0.84` to `0.85`.

The 2026-06-14 UTC batch keeps the direct averaging rollback in force and spends quota on untested structured anchors plus independent public model adapters. The Keith no-experts, Mamba, and non-alpha reciprocal anchors test controlled ablations inside the 0.86-class anchor family. The completed results did not improve the public best: the no-experts anchor matched `0.86`, Mamba and non-alpha reciprocal scored `0.85`, Ramkan scored `0.85`, and Quincy scored `0.78`. The Kaichengyu and Priya wrappers errored before creating competition submissions because Kaggle did not expose a valid adapter pair to the wrapper despite file manifests suggesting adapter artifacts.

The 2026-06-15 UTC batch uses the 2026-06-14 feedback to avoid spending all quota on the same 0.86-class anchor family. The selected set includes one QR-SVD fusion retry, one TIES merge candidate that is more structured than the failed naive average, one independent ATLAS solver-augmented SFT output, one AGI/Huikang SVD control, and one recent Mirza artifact check. This mix preserves a high-score-line control while reserving most quota for method-diverse or source-diverse candidates. The final completed scores identify `mirza_lora_safetensors_output` as the selected private leaderboard candidate with public `0.86` and private `0.86`.

The final selection for private leaderboard use prefers a candidate that combines strong public score, valid packaging, and acceptable private leaderboard stability. The final reported outcome is Silver Medal, private leaderboard rank `167`, and private score `0.86`.

## Stop Condition

The Kaggle competition metadata reports `maxDailySubmissions = 5`. On 2026-06-15 UTC, the submission table contains five recorded submissions and the quota check reports `remaining = 0`, so the daily quota is exhausted.
