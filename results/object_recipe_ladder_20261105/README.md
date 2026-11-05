# Completed object recipe ladder, 2026-11-05

The A/B/C and D1-D3 fits and their fixed-final validation evaluations are complete. The registered retention rule selects **D2**. This snapshot uses training seed 1247. Formal multiple-training-seed comparison and held-out test scoring have not run.

## Aggregate validation results

MAE in degrees, lower is better. The primary GSO validation panel contains 93 objects, four views per angle, and five angles, totaling 1,860 images. The two lighting panels contain the same 93 GSO validation objects at 30 degrees, with 372 images each. They are GSO lighting controls. DiLiGenT and LUCES use one fixed lowest-ID light per object, 10 and 14 images respectively.

| Model | GSO five-angle | GSO original-light 30deg | GSO Arb-style-light 30deg | DiLiGenT single-light | LUCES single-light |
|---|---:|---:|---:|---:|---:|
| A | 24.0520 | 23.3342 | 23.9808 | 22.4963 | 21.2210 |
| B | 24.3220 | 23.9843 | 24.2999 | 24.4226 | 20.9167 |
| C | 22.1420 | 21.5386 | 22.2453 | 22.4524 | 20.5657 |
| D1 | 21.7151 | 21.0851 | 22.1662 | 22.8644 | 20.6946 |
| D2 | 18.6650 | 18.1865 | 18.9917 | 18.2309 | 17.7026 |
| D3 | 19.0845 | 18.5342 | 19.3571 | 18.7242 | 18.7892 |
| NeuralLightRig | 20.5024 | 20.4653 | 23.7794 | 23.2198 | 25.1926 |

The fresh-stream Neural LightRig row uses one seed-511 RNG sequence and the recorded per-image state replay for GSO angle and lighting controls. Its real-domain stream differs from the legacy-order reruns below. These real scores use one image per object. The RoSE ten-light protocol uses a different selection. The earlier 2026-10-24 snapshot remains a separate historical run using the historical 52,500-update R/V/Xn checkpoints and its own rendered inputs and RNG protocol.

Our Xn scores average errors over image-specific inference seeds 42-49. This is not vector ensembling. B is deterministic R and carries one seed-42 label. Our default UNet input is 256, D3 input is 512, and Neural LightRig input is 512 with official 75-step inference. Native GT, masks, normalization, and model-specific input resizing follow the published code. Current aggregate details are in [summary.json](summary.json), and the frozen measurement conditions are in [protocol.json](protocol.json).

## Primary angle breakdown

| Model | 0deg | 15deg | 30deg | 45deg | 60deg |
|---|---:|---:|---:|---:|---:|
| C | 20.3858 | 21.4530 | 22.0535 | 23.1556 | 23.6621 |
| D1 | 19.5213 | 20.8246 | 21.9213 | 23.0244 | 23.2836 |
| D2 | 17.9390 | 17.8982 | 18.5581 | 19.3395 | 19.5900 |
| D3 | 18.0386 | 18.0872 | 18.9174 | 20.0268 | 20.3527 |
| NeuralLightRig | 18.5732 | 20.8252 | 20.6658 | 20.6177 | 21.8299 |

## Actual training budgets

| Model | Training assets | Prepared views | UNet side | Updates | Objective |
|---|---:|---:|---:|---:|---|
| A | 1,780 | 5,335 | 256 | 50,000 | Xn |
| B | 1,780 | 5,335 | 256 | 50,000 | R |
| C | 10,000 | 29,976 | 256 | 50,000 | Xn |
| D1 | 40,000 | 239,816 | 256 | 100,000 | Xn |
| D2 | 10,000 | 29,976 | 256 | 50,000 | Xn |
| D3 | 1,038 | 3,111 | 512 | 10,000 | Xn |

A/B/C/D1 use a frozen DINOv2-S prior. D2 uses the C dataset, budget, and fresh controlled initialization with a frozen DINOv2-B prior at canonical 518 input. D1 uses six fixed views and all seven lights, 239,816 valid views and 1,678,712 cached RGBs. D3 warmstarts D2 and uses the hash-selected native subset intersected with the retained C dataset, yielding 1,038 assets and 3,111 views. Its views and light selection follow the retained recipe. Its dataset size, resolution, learning rate, and initialization differ from D2. The published configuration records these conditions. The comparison therefore measures this combined native-subset fine-tuning configuration.

## Registered retention decisions

Retain when the previous retained recipe MAE minus the candidate MAE is at least 0.5 degrees, using unrounded values. Each stage uses its fixed final checkpoint.

| Candidate | Previous retained recipe | Improvement, degrees | Decision |
|---|---|---:|---|
| D1 | C | +0.4269 | revert |
| D2 | C | +3.4770 | retain |
| D3 | D2 | -0.4196 | revert |

The final retained recipe is D2, MAE 18.66496716 degrees. Its checkpoint SHA256 is `f1cbe7e14db11c4856677731c6294d3a55b4068dcbd2f59fecae68e9fa05d170`. The configuration is [D2_actual_20261030.json](../../configs/D2_actual_20261030.json).

## Neural LightRig legacy-order real-domain reruns

The model is loaded first. Each domain then resets seeds 511-515 and sorts by SHA256 image ID according to the legacy-order rule. The five values are inference RNG replicates. They do not measure training-seed uncertainty. All fixed one-light inputs are retained.

| Dataset | Images per seed | Mean MAE | Sample SD, ddof=1 | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| diligent | 10 | 26.9448 | 2.2813 | 23.2715 | 28.9225 |
| luces | 14 | 21.9060 | 2.4142 | 19.8691 | 26.0619 |

These are separate RNG protocols from the fresh-stream row in the first table. Per-seed aggregate MAEs and the scope are included in the JSON snapshot.

## C inference ablations

| Inference | GSO five-angle | DiLiGenT | LUCES |
|---|---:|---:|---:|
| single42 | 22.1516 | 22.5542 | 20.4851 |
| flip_TTA42 | 21.5635 | 22.1034 | 20.3679 |
| mean4 | 21.9788 | 22.2740 | 20.4465 |
| flip_TTA_mean4 | 21.4319 | 21.9293 | 20.3491 |
| C_original8_error_mean_reference | 22.1420 | 22.4524 | 20.5657 |

`mean4` averages predicted unit vectors from four samples and renormalizes. Flip variants invert the spatial reflection and normal X component before averaging. The original eight-seed reference averages eight errors. It is not an eight-vector ensemble. The registered single-42 control drift is 0.0. TTA and ensembling are reported separately and do not enter the D-stage retention gate.

## Reconciliation and execution

Independent reconciliation read 110,376 private metric records for A/B/C, Neural LightRig, and D1-D3. All five panels and their per-pitch object-macro MAEs agree with the saved summaries within 1e-9 degrees. The observed maximum absolute difference is 1.42e-14 degrees. All three retention decisions and the final recipe were recomputed. Aggregate checks are in [validation.json](validation.json).

Obtain source data, pinned DINO and Neural LightRig components, and checkpoints separately. Execute from the repository root with an appropriate training environment. The published scripts retain source/configuration registration checks. Preparing private manifests and creating matching local registration receipts are prerequisites. The publication manifest records source-side and exported hashes because path normalization changes exported bytes. Source configuration hashes describe the completed local runs.

For C and D1 training use `scripts/train_object_dino_C_20261029.py --group C` and `scripts/train_object_dino_D1_20261030.py --group D1`. For D2/D3 set `FACEFLUX_STAGE_CONFIG` to the applicable published configuration and run `scripts/train_D_stage_20261030.py --group D2` or `--group D3`. The matching evaluator is `scripts/evaluate_D_stage_20261030.py`. All evaluation commands require locally prepared registrations and frozen validation inputs. Run `python -m pytest -q tests` for the repository checks.

Data, masks, images, predictions, weights, sample lists, per-image records, private registration receipts, machine inventory, and research interpretations are kept local under the established publication policy. Results are conditional on this validation selection and one training seed. No formal test images have been scored.
