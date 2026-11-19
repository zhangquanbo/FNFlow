# ICME validation reproduction snapshot

The source snapshot adds the completed E-round model/training entries and the later S1/B1/ablation/metric/latency code. Machine-specific roots are normalized; original and exported hashes are distinct. Aggregate tables can be inspected without data or weights. Training and image evaluation require externally supplied data, private split manifests, teacher receipts and pretrained weights; those prerequisites are not distributed. No new dataset, weight or software-license grant is implied.

Recorded runtimes: training PyTorch2.1.2/CUDA12.1 with BF16; evaluation/timing PyTorch2.4.1/CUDA12.1. Timing GPU RTX3090. Python, NumPy, Pillow, OpenCV, SciPy and torchvision must be compatible with the chosen Torch installation. Install the existing requirements before supplying the pinned DINO source archive and the MoGe backbones/teacher cache referenced in configuration.

## Fresh S1 fit with external prerequisites already present

From the repository root, on a BF16-capable CUDA device:

```bash
python scripts/register_icme_reproduction.py --config configs/icme2027_20261119/S1_s1247.json
CUDA_VISIBLE_DEVICES=0 python scripts/train_ICME_Rplus_20261110.py --config configs/icme2027_20261119/S1_s1247.json --smoke --reference
CUDA_VISIBLE_DEVICES=0 python scripts/train_ICME_Rplus_20261110.py --config configs/icme2027_20261119/S1_s1247.json
```

The trainer saves all optimizer/EMA/Python/NumPy/Torch/CUDA RNG states and can resume with `--resume`. At25000 it stops for a diagnostic. For these locked follow-ups the configured threshold is null: run the25000 evaluator and resume to50000; never select the best checkpoint. A diagnostic does not authorize scoring GSO-test.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/ICME_locked_Rplus_evaluate_20261112.py --config configs/icme2027_20261119/S1_s1247.json --stage 25000
CUDA_VISIBLE_DEVICES=0 python scripts/train_ICME_Rplus_20261110.py --config configs/icme2027_20261119/S1_s1247.json --resume
CUDA_VISIBLE_DEVICES=0 python scripts/ICME_locked_Rplus_evaluate_20261112.py --config configs/icme2027_20261119/S1_s1247.json --stage 50000
```

Use the analogous1248/1249/no-distillation configurations for independent runs. S0 uses the legacy frozen-prior entry and its group label is historical naming, not the backbone size:

```bash
python scripts/register_icme_reproduction.py --config configs/icme2027_20261119/S0_s1247.json --trainer scripts/train_ICME_E1_replica_20261110.py
FACEFLUX_STAGE_CONFIG=configs/icme2027_20261119/S0_s1247.json CUDA_VISIBLE_DEVICES=0 python scripts/train_ICME_E1_replica_20261110.py --group E1_B
```

S0 likewise has a25000 diagnostic handled by `ICME_locked_S0_evaluate_20261112.py`. The historical timing workers retain their registered run layouts and require adapting local checkpoint/environment paths to a fresh reproduction; they are not a dataset downloader. The idle latency case refuses training overlap and includes process-level CPU/GPU telemetry. GPU affinity and compiler flags are part of the measured variant.

## Scope checks

The publication check parses exported Python/JSON, checks local module dependencies, reconciles all55 new accuracy table cells to their source aggregates, recomputes three-training-seed mean/sample SD, checks the original source/config signatures, and scans only the explicit allowlist for excluded sample identifiers, machine paths and credentials. It does not claim a new GPU training rerun or a full optimized-model accuracy evaluation. See the public validation JSON and source/export hashes. Existing historical result directories remain unchanged.
