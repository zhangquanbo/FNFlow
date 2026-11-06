# MoGe-2 S/B/L and D2 accuracy, 2026-11-06

All four models completed all five panels in the same PyTorch2.4.1 environment. Every D2 cell is a fresh current-backend measurement, including GSO. No old frozen-reference cell is mixed into this completed table.

| Model | GSO5 pitches | DiLiGenT1 | LUCES1 | DiLiGenT10 | LUCES10 | End-to-end ms |
|---|---:|---:|---:|---:|---:|---:|
| D2 | 18.6588 | 18.2167 | 17.6614 | 18.0599 | 17.0936 | 83.55 |
| MoGe-S | 19.4540 | 19.0669 | 17.7428 | 18.0706 | 15.9987 | 84.42 |
| MoGe-B | 16.1116 | 16.5603 | 15.8885 | 15.5833 | 14.3758 | 92.21 |
| MoGe-L | 13.7934 | 13.6061 | 13.3924 | 13.4427 | 11.9216 | 146.58 |

Errors are angular degrees; lower is better. GSO uses93 validation objects,4 views and5 pitches (1,860 images), with equal object/view means per pitch then equal5pitch means. Real controls contain10 DiLiGenT and14 LUCES objects. Ten-light panels contain100 and140 independent monocular inputs, averaging errors within each object then objects equally. This does not denote joint multi-image inference.

D2 is BF16; MoGe is FP16. The [matched3090 timing report](../moge2_scale_confirmation_20261106/README.md) also contains model-only, full-official and FP32 timings. Model loading is excluded; PNG read, preprocessing/transfer, prediction and common256-normal PNG write are included. D2 latency is one prediction; its accuracy is the mean error of8 independent inference draws, not an8-vector ensemble. MoGe normal-only output exactly matched full official FP16 normal output on9 independent equivalence cases.

MoGe S/B/L have35,103,656 /104,743,592 /330,901,544 parameters. All8,496 metric rows passed exact frozen input-ID sets, image/object/pitch bindings, registration signatures and independent macro aggregation checks. No failed image was omitted. [validation.json](validation.json) contains aggregate validation only. [summary.json](summary.json) provides pitch aggregates and separate latency/error and parameter/error Pareto comparisons.

MoGe-L GSO is13.793366deg in this rerun versus13.793539deg in the previous full-inference snapshot. Source, precision and backend settings are retained in [protocol.json](protocol.json). The [data/license audit](../moge2_scale_confirmation_20261106/data_and_license.json) documents public-source limitations. GSO-test remains sealed; the deferred formal3-training-seed comparison is not asserted complete.
