# MoGe-2 scale confirmation, 2026-11-06

Completed: pinned S/B/L normal checkpoint loading, source/license disclosure, and matched RTX 3090 timing. The previous accuracy evaluation used **Ruicheng/moge-2-vitl-normal**, revision cb0e8bbd6b1e243589717c78e750b1ba4c093acf. New three-size accuracy measurements are separate and remain in progress at this snapshot.

| Model | Total parameters | Compute precision | Normal-path end-to-end median (ms) | Model-only median (ms) |
|---|---:|---|---:|---:|
| D2 | 130,400,259 | bfloat16 | 83.55 | 37.61 |
| MoGe-S | 35,103,656 | float16 | 84.42 | 33.52 |
| MoGe-B | 104,743,592 | float16 | 92.21 | 50.43 |
| MoGe-L | 330,901,544 | float16 | 146.58 | 100.02 |

End-to-end: one PNG read, configured preprocessing and transfer, one prediction, conversion to the common normal frame, bilinear resize and unit normalization at 256, and one normal PNG write. Model loading is excluded. Each row has 5 warmups and 30 synchronized timed trials. All models use the same PyTorch 2.4.1 CUDA 12.1 environment; original D2 BF16 and MoGe FP16 compute are explicitly distinguished. The input is the same 512-square file. D2 uses a 256 core and a 518-square prior (1,369 tokens); MoGe uses resolution level 9 and 3,600 tokens (840-square encoder input). These input settings are not equalized and are recorded rather than hidden.

The primary MoGe path runs its normal encoder, neck and head, omitting unused geometry heads and focal/shift recovery. For each size, three independent non-benchmark RGB examples (square, landscape and portrait) produced exactly the same raw FP16 normal tensor as full official inference: maximum absolute difference 0. Full official inference and FP32 timings are included separately in [efficiency.json](efficiency.json); those diagnostic times must not be paired with FP16 accuracy as if they were the same mode.

Official-full-inference FP16 medians are MoGe-S 106.44 ms, MoGe-B 114.52 ms, MoGe-L 169.53 ms.

The public MoGe training configuration lists ObjaverseV1 among 24 datasets, but does not list GSO, DiLiGenT or LUCES. This does **not** establish whether specific evaluation assets/images overlapped pretraining. An exact checkpoint training-instance list was not found in the audited public material. Our Arb-Objaverse training also shares the Objaverse source. The normal head was added after the original paper, so the paper alone is not an exact checkpoint training manifest. See [data_and_license.json](data_and_license.json) for sources and limitations.

Code is MIT, vendored DINOv2 is Apache 2.0, and all three Hugging Face normal-model cards declare MIT. Checkpoint repositories, revisions, full SHA256 values, inference settings and source hashes are in [protocol.json](protocol.json). No checkpoints, input IDs, arrays, per-image metrics, timing-sample arrays or machine inventories are included. GSO-test remains sealed.

The completed normal-inference helper is [src/moge_bridge_20261106.py](../../src/moge_bridge_20261106.py), with the public checkpoint registry in [configs/moge2_normal_models_20261106.json](../../configs/moge2_normal_models_20261106.json). Its only export change replaces the private registration filename with that public registry. Place the pinned upstream source under `external_baselines_20261030/MoGe2/source`, and downloaded model files at the relative registry paths. SHA256 verification is mandatory. Model code derives from Microsoft MoGe; see [the upstream MIT license](../../licenses/MoGe-MIT.txt). Private benchmark inputs and trained D2 weights are not included.
