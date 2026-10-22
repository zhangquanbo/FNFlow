# FNFlow

Pixel-space conditional Flow Matching for face normals, with isolated research code for general object and indoor-scene normal estimation.

This repository contains the historical UNet baseline, experiment code and aggregate numerical results. **No new general-domain model has been trained yet.** Future public updates are limited to code/configuration, operational instructions and aggregate results.

## Contents

- [Historical baseline](baseline/)
- [Full results](results/D1_full/summary.json)
- [Pilot results](results/P0_D1_D5_pilot/summary.json)
- [Full eight-seed results](results/D2_D3_full_multiseed/multiseed_summary.json)
- [Publication scope](PUBLICATION.md)

## First diagnostic results

UNet-B has 43,081,539 parameters. Full FaceScape diagnostics use 1,998 images and the historical face-mask protocol. The original best result is reproduced at **17.899513816 degrees**. These are historical development-domain results, not object/scene generalization results.

| Training checkpoint | Conditional | Fixed CFG 1.5 | Geom-CFG |
|---|---:|---:|---:|
| dropout 0.3 | 19.7527 | 21.6212 | 19.8666 |
| dropout 0.5, original | 17.9906 | 19.3185 | 17.9513 |
| dropout 0.7 | 25.6053 | 29.2225 | 27.3178 |

Mean angular error in degrees; lower is better. A checkpoint trained without condition dropout is evaluated only conditionally.

![Full CFG diagnostics](figures/D1_full.png)

Step-count, eight-seed and averaging diagnostics use a fixed **32-identity pilot**, not all 1,998 images. They show remaining noise dependence and do not establish calibrated uncertainty.

![Pilot diagnostics](figures/D2_D4_pilot.png)

## Full eight-seed follow-up

The follow-up uses the fixed p05_best checkpoint, all 1,998 historical images and inference seeds 42–49. Each score is first averaged over images, then over seeds. SD is the sample SD of the eight dataset MAEs (ddof=1), not training-seed variation or a generalization confidence interval. Identical initial fields are paired across step counts; the historical evaluator resets a seed per image.

| Inference | Euler steps | Float-normal MAE mean ± seed SD, degrees |
|---|---:|---:|
| Geom-CFG | 1 | 19.2632 ± 1.0382 |
| Conditional | 1 | 19.5688 ± 1.1633 |
| Conditional | 2 | 19.8400 ± 1.4971 |
| Conditional | 4 | 20.2608 ± 1.9732 |
| Conditional | 8 | 20.5925 ± 2.3104 |

All 40 configurations completed. Per-seed aggregate scores and paired step differences are in the linked JSON. No sample identifiers or per-image results are added by this follow-up release.

## Image-specific noise follow-up

Same fixed checkpoint and full 1,998-image set, one Euler step, eight inference replicates. A versioned SHA256 of the base seed and dataset-relative RGB key derives a different initial field for each image. The fields are paired across guidance modes; the historical shared-field results above are retained.

| Inference | MAE mean ± dataset-replicate SD, degrees | Mean within-image MAE SD, degrees | Mean pixel angular dispersion, degrees |
|---|---:|---:|---:|
| no_cfg | 19.8371 ± 0.0199 | 1.3500 | 3.3175 |
| geom_cfg | 19.4122 ± 0.0202 | 1.3228 | 4.0873 |

The dataset-replicate SD measures Monte Carlo variation of the dataset mean. Within-image MAE SD and pixel directional dispersion measure noise sensitivity; they are distinct statistics. Displayed SDs use ddof=1. None is training-seed SD or a generalization confidence interval. Pixel dispersion is the mean angle to the directional sample mean within each evaluation mask, then macro-averaged over images.

[Aggregate results](results/D3_full_imagewise_noise_v1/multiseed_summary.json) include the shared-field reference and the mean directional variance 1-R. Sample keys, seed mappings and per-image records are not exported.

## Directional averaging and identity resampling

Same full historical set and image-specific noise. Existing checkpoints are kept separate; neither was retrained.

| Checkpoint | Conditional MAE° | Geom-CFG MAE° | Geom-CFG ensemble of 8 MAE° | Single-sample Geom-CFG identity interval° |
|---|---:|---:|---:|---|
| p05_best | 19.8371 | 19.4122 | 18.9568 | [19.1502, 19.6764] |
| p05_original | 19.1481 | 19.2544 | 18.7597 | [18.9852, 19.5327] |

Intervals are 95% percentile cluster-bootstrap intervals: 100 identities resampled 10,000 times, conditional on the fixed weights, inference draws and development-domain identity population. Whole clusters are sampled and the macro-image ratio is recomputed. They do not include training-seed variation or establish generalization to new domains. Eight-sample directional averaging costs eight samples and changes the estimator; its gain is not an isolated irreducible noise cost.

[Aggregate results and paired intervals](results/D4_D6_face_supplements_v1/aggregate_summary.json).

## Data preparation status

The existing Hypersim conversion contains 7,325 samples; 3,146 images have at least one zero normal inside the old valid mask. A new native-geometry FP32 cache contains 5,736 train and 700 validation images from 30/5 disjoint scenes. All prepared samples pass the included loader checks. Exact camera projection matrices are retained; an ordinary K has a separate validity flag. RGB comes from lossy final-preview images, not the original HDR color.

Mixed-domain training is not ready. Official NYUv2 scene metadata and train/test indices are now checked: 795 images from 249 rooms are reserved for real-indoor validation, with no room overlap with the 654-image test partition. Three non-test RGB arrays are not byte-identical to the packaged images; preprocessing correspondence remains under investigation. A pinned Arb-Objaverse sample contains 3 assets, 6 geometry views and 12 RGBA inputs. It has alpha masks but no explicit depth or intrinsics; the Cycles normal-pass semantics require a separate contract from pure geometry normals. The 300 DSINE ScanNet benchmark frames and 42 OLATverse validation objects remain reserved. Data, sample manifests and prepared caches are not distributed. [Inventory](results/P0_general_data_v1/summary.json), [prepared-cache checks](results/P0_prepared_hypersim_full_v1/summary.json), [object evaluation inventory](results/P0_object_benchmark_v1/summary.json), [Arb sample checks](results/P0_arb_protocol_v1/summary.json), [NYUv2 mapping checks](results/P0_nyu_validation_v1/summary.json).


## Setup

Tested on Linux, Python 3.10.19, PyTorch 2.1.2 + CUDA 12.1, torchvision 0.16.2 and RTX 4090 (24 GB). Observed versions are in [tested_environment.json](reproducibility/tested_environment.json).

Install an appropriate PyTorch build for your hardware using requirements-torch.txt, then the recorded dependencies in requirements.txt. dlib may require compiler/CMake support.

Datasets, weights and the landmark predictor are **not distributed**. Obtain them separately with the relevant permissions. Set a read-only reference directory:

~~~bash
export FACEFLUX_BEST_ROOT=/path/to/FaceFlux-Best
export PYTHONDONTWRITEBYTECODE=1
~~~

Expected checkpoint and data paths are declared in [diagnostics_pilot.json](configs/diagnostics_pilot.json).

## Run

From the repository root:

~~~bash
python -m pytest -q tests -o cache_dir=outputs/.pytest_cache
python scripts/preflight.py
python scripts/run_diagnostics.py
python scripts/run_diagnostics.py --stage d1 --subjects 0 --run-id D1_full
python scripts/run_multiseed_diagnostics.py
python scripts/run_imagewise_noise.py
python scripts/verify_best.py
python scripts/plot_diagnostics.py
python scripts/summarize_diagnostics.py
python scripts/render_record.py
~~~

Published snapshots are in results/; new runs write outputs/. The original preflight audit is intentionally not overwritten. Reusing a run ID requires identical code/configuration/sample selection.

Report scripts regenerate workspace-oriented documents. This publication export normalizes machine paths and document links.

## Scope

baseline/ preserves the historical face pipeline and checkpoint-zoo configurations. Its command snapshots require local path adaptation; they are not the new general training entry point.

The source and output protocols are covered by the included tests. Full-set and pilot measurements are distinct and should not be pooled.

No dataset, pretrained-weight license or new software-license grant is implied by repository visibility. See [NOTICE.md](NOTICE.md).
