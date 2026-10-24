# Completed object normal comparison

Completed inference-only evaluation of Neural LightRig and existing52,500-update R/V/Xn checkpoints. No new training. MAE degrees, lower is better; object macro.

| Dataset (objects/images) | Neural LightRig | R | V | Xn |
|---|---:|---:|---:|---:|
| gso (198/792) | 20.2410 | 30.4605 | 30.9454 | 29.9913 |
| arb (128/384) | 29.9299 | 25.6840 | 28.1681 | 28.1133 |
| diligent (10/10) | 28.9285 | 26.7426 | 25.9474 | 27.6081 |
| luces (14/14) | 21.7068 | 25.1059 | 26.1473 | 24.7425 |

[Full aggregate JSON](summary.json) includes both macro aggregations, main/secondary GT, strict pixel thresholds, boundary coverage, inference-seed sensitivity, paired-object counts, GT disagreement and efficiency. [Pinned protocol and component hashes](../../configs/object_comparison_release_20261024.json).

## Measurement conditions

- Seeds averaged within each image metric, then image/object macro. Median is macro of within-image pixel medians; thresholds are strict pixel percentages, not percentages of objects.
- GT adjacent normal jump >20 degrees, radius2 dilation and GT-valid intersection; excludes silhouette unless it also meets the angular definition. Empty regions are null with coverage counts.
- R deterministic; V/Xn image-specific CUDA noise seeds42..49. NLR default511 once per sorted input stream. Pre-registered NLR subset rerun seeds511/512/513; subset511 does not share the full-stream RNG order.
- One preselected lowest-ID light per object, not the RoSE ten-image21..30 protocol. No per-image exposure, gamma, light-intensity division or crop. uint16/65535 mapped to8bit. Official masks. Pot2 excludes73 zero-GT official-mask pixels, before scoring; no LUCES exclusions.
- Shared4090,512x512,batch1,5warmups,30timings. OfficialNLR float32; ours bf16 autocast with float32 normalization. Accuracy inference ours256/NLR512. Native output-writing paths differ; not an identical-precision pure-compute comparison.
- GSO main smooth geometry, secondary face and crease45. Arb retains BSDF normal-pass labels and its GT-valid contract. UID exclusion does not prove exhaustive cross-source geometry deduplication.
- Inference seed variation only; one training seed for R/V/Xn; not training-seed uncertainty or confidence intervals.
- Aggregates and code only. Source images, private manifests/records, predictions, example figures, external weights and research interpretations are not distributed.

RoSE v5 Table1/2 reference: Neural LightRig29.10 degrees on DiLiGenT and20.95 degrees on LUCES, using ten lights21..30 per object. This release uses one lowest-ID light. Source: https://arxiv.org/pdf/2602.09929v5 . Numerical proximity is not proof of exact protocol reproduction.

## Local execution

Use an independent Python3.11+/PyTorch2.4+ environment for Neural LightRig. Obtain the pinned upstream source/weights and base components with your own access, and configure the offline paths expected by src/lightrig_adapter.py. The original training environment is separate.

The preparers require local source datasets and registered protocol receipts. They generate private manifests, not a distributed test list. After preparing your own manifests, scripts/evaluate_object_comparison.py uses OBJECT_COMPARISON_MANIFEST and OBJECT_COMPARISON_OUTPUT to select them. scripts/validate_object_comparison_results.py independently checks complete local results. Inference is serialized by the shared GPU lock. These commands do not train models.

The original masks, missing/invalid-GT counts, input range mapping and coordinate transforms are fixed before scoring. RGB compositing differs only as declared: transparent alpha for NLR, white background for ours.

The original archived WSD training commands/results remain historical; this object evaluation does not authorize or start that training queue. External datasets, official example GT (unavailable for the clay vase), weights and sample images are not supplied.
