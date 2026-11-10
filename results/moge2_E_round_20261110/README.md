# Completed MoGe-2 prior and normal-refinement experiments

Completed 2026-11-10. Six new fits, each training seed1247 and fixed final50,000 successful optimizer updates. All five panels and RTX3090 forward/throughput measurements are complete. These are single-training-seed validation results; GSO-test was not evaluated.

## Effective experiment conditions

| Experiment | Executed condition |
|---|---|
| E1_B | FM with frozen MoGe-B geometry encoder |
| E2_L | FM residual refinement conditioned on MoGe-L normals |
| E3_L | MoGe-L normal-head-only fine-tuning |
| E2_L_direct | D2 direct-output head with added MoGe-L normal input |
| E2_B | FM residual refinement conditioned on MoGe-B normals |
| E1_L | FM with frozen MoGe-L geometry encoder |

All six use10,000 training assets/29,976 images and effective batch16 (800,000 sample exposures). FM uses256 output, canonical518 frozen DINO features and LR2e-4. E3 uses native512 RGB/256 GT, LR1e-4, FP16 normal-head-only fine-tuning; its physical batch1 x16 accumulation is grouped into four logical batches of4. See [recipes.json](recipes.json) for executed settings.

E2-B/L retains the original frozen DINOv2-B. Relative to D2 it adds3 normal-conditioning components and changes the output to coarse normal + zero-initialized learned residual. The direct ablation retains the original D2 output/initialization and adds only the3 conditioning inputs. The initial residual model equals the processed256 condition, not necessarily native MoGe-L at real GT resolution; [processing_control.json](processing_control.json) quantifies this processing effect.

## Angular MAE (degrees, lower is better)

| Model | GSO-val | DiLiGenT single | LUCES single | DiLiGenT10-light | LUCES10-light |
|---|---:|---:|---:|---:|---:|
| D2 | 18.6588 | 18.2167 | 17.6614 | 18.0599 | 17.0936 |
| MoGe-S | 19.4540 | 19.0669 | 17.7428 | 18.0706 | 15.9987 |
| MoGe-B | 16.1116 | 16.5603 | 15.8885 | 15.5833 | 14.3758 |
| MoGe-L | 13.7934 | 13.6061 | 13.3924 | 13.4427 | 11.9216 |
| E1_B | 17.5134 | 17.9769 | 17.4741 | 17.9910 | 16.6677 |
| E2_L | 14.1987 | 13.9056 | 13.5493 | 13.8124 | 12.1964 |
| E3_L | 14.3630 | 14.3908 | 13.3816 | 14.3514 | 12.2734 |
| E2_L_direct | 14.5053 | 13.9450 | 13.8060 | 13.9354 | 12.5420 |
| E2_B | 16.3089 | 16.6318 | 16.1226 | 15.9281 | 14.9482 |
| E1_L | 15.5369 | 16.4606 | 15.3864 | 16.3995 | 14.9384 |

GSO-val:93 objects x4 views x5 pitch bins,1860 images. Real single:10/14 images.10-light:100/140 independent monocular predictions, averaged within object and then across objects; no joint10-input inference. FM numbers average8 independent error measurements(seeds42-49), not8 training seeds or vector-averaged predictions. Native GT resolution and masks are preserved.

## RTX3090 pure forward and throughput

| Model | Parameters(M) | Batch1 wall median(ms) | Batch4 images/s |
|---|---:|---:|---:|
| D2 | 130.400 | 37.891 | 30.684 |
| MoGe-S | 35.104 | 33.915 | 32.889 |
| MoGe-B | 104.744 | 50.925 | 22.817 |
| MoGe-L | 330.902 | 100.172 | 11.047 |
| E1_B | 130.400 | 38.080 | 30.548 |
| E2_L | 461.304 | 136.920 | 8.110 |
| E3_L | 330.902 | 99.711 | 11.052 |
| E2_L_direct | 461.304 | 136.968 | 8.125 |
| E2_B | 235.146 | 95.429 | 13.126 |
| E1_L | 348.434 | 97.611 | 14.097 |

FM BF16, MoGe/E3 FP16; parameters stored FP32. Same exclusive RTX3090/PyTorch2.4.1+cu121; TF32 off,5 warmups and30 timed repeats. Inputs, noise and timesteps are already on GPU. No fileIO, host transfer, PNG processing or random generation. E2 includes online MoGe inference plus refinement. Full batches1/2/4/8, CUDA-event medians, p90s and peak allocated memory are in [efficiency.json](efficiency.json). These numbers must not be mixed with earlier read/write-inclusive E2E measurements.

## GSO-val pitch and primary-boundary diagnostics

| Model | 0deg | 15deg | 30deg | 45deg | 60deg | Boundary MAE |
|---|---:|---:|---:|---:|---:|---:|
| D2 | 17.9357 | 17.8891 | 18.5475 | 19.3394 | 19.5824 | 22.2578 |
| MoGe-S | 16.9429 | 17.4849 | 18.7394 | 20.7937 | 23.3094 | 22.9795 |
| MoGe-B | 15.2799 | 14.5022 | 15.1593 | 16.6741 | 18.9423 | 19.4975 |
| MoGe-L | 13.9457 | 12.7276 | 13.0472 | 13.8324 | 15.4140 | 17.0962 |
| E1_B | 17.0017 | 16.3370 | 17.0461 | 18.0249 | 19.1574 | 21.4864 |
| E2_L | 14.6185 | 13.4307 | 13.7589 | 14.0734 | 15.1121 | 17.4959 |
| E3_L | 14.7813 | 13.9608 | 13.9799 | 14.0446 | 15.0484 | 17.8957 |
| E2_L_direct | 14.8745 | 13.6726 | 14.0555 | 14.5742 | 15.3495 | 17.8486 |
| E2_B | 16.0812 | 15.2426 | 15.6577 | 16.5348 | 18.0283 | 19.8196 |
| E1_L | 15.5745 | 14.8745 | 15.0385 | 15.6483 | 16.5486 | 19.4885 |

Primary boundary: GT adjacent normal angle>20deg, radius2 dilation, intersected with valid mask. Empty boundary bands are excluded only from that boundary metric, with the same frozen masks across methods. [boundary_and_pitch.json](boundary_and_pitch.json) also includes outside-boundary errors, alternate GT diagnostics and pixel-pooled upward-region Z bias per pitch.

## Audit and available files

Recomputed21,240 private metric rows:12,744 new-fit rows and8,496 reference rows. Exact frozen manifest ID sets, object/view identities, registration signatures, finite metrics, source/config hashes and final checkpoint SHA256 values were checked. Aggregate reconciliation tolerance1e-10deg. E3 logged13 FP16 overflow retries of the same batch, while completing50,000 successful updates; frozen state was verified unchanged for all fits.

- [summary.json](summary.json): full-precision accuracy and training wall times.
- [accuracy.csv](accuracy.csv), [efficiency.csv](efficiency.csv): aggregate tables.
- [protocol.json](protocol.json), [recipes.json](recipes.json): evaluation/timing definitions and effective training conditions.
- [validation.json](validation.json), [provenance.json](provenance.json): validation counts and source/config/checkpoint hashes.
- [processing_control.json](processing_control.json): zero-residual input-processing control.

Reference model cards, pinned revisions, training-data audit and licenses remain in [the scale confirmation snapshot](../moge2_scale_confirmation_20261106/README.md). The training-data list includes ObjaverseV1; asset-level membership is not public, so overlap cannot be established or excluded from dataset names alone. GSO-val has been used for recipe comparison; these are not sealed test results. No formal three-training-seed comparison or full G-buffer training is included in this completed round.
