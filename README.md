# FNFlow

Pixel-space conditional Flow Matching for face normals, with isolated research code for general object and indoor-scene normal estimation.

This repository contains the historical UNet baseline, research code, revised plan and numerical results from the first diagnostic experiments. **No new general-domain model has been trained yet.**

## Contents

- [Research plan](docs/研究方案.md)
- [Code and experiment record](docs/代码与实验记录.md)
- [First diagnostic findings](docs/首轮诊断结论.md)
- [Dataset source audit](docs/数据来源初查.md)
- [Historical baseline](baseline/)
- [Full results](results/D1_full/summary.json)
- [Pilot results](results/P0_D1_D5_pilot/summary.json)
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
python scripts/verify_best.py
python scripts/plot_diagnostics.py
python scripts/summarize_diagnostics.py
python scripts/render_record.py
~~~

Published snapshots are in results/; new runs write outputs/. The original preflight audit is intentionally not overwritten. Reusing a run ID requires identical code/configuration/sample selection.

Report scripts regenerate workspace-oriented documents. This publication export normalizes machine paths and document links.

## Scope

baseline/ preserves the historical face pipeline and checkpoint-zoo configurations. Its command snapshots require local path adaptation; they are not the new general training entry point.

There are 13 passing protocol tests. The research direction remains a hypothesis: reliable geometric detail under mixed-quality supervision, tested against matched regression/FM controls.

No dataset, pretrained-weight license or new software-license grant is implied by repository visibility. See [NOTICE.md](NOTICE.md).
