# MoGe-2 S/B/L normal accuracy, 2026-11-06

All three MoGe scales completed all five panels. D2 real-data numbers were recomputed in the same PyTorch2.4.1 environment. D2 GSO marked * is the previously published frozen validation reference; its current-environment control is still running.

| Model | GSO5 pitches | DiLiGenT1 | LUCES1 | DiLiGenT10 | LUCES10 | End-to-end ms |
|---|---:|---:|---:|---:|---:|---:|
| D2 | 18.6650* | 18.2167 | 17.6614 | 18.0599 | 17.0936 | 83.55 |
| MoGe-S | 19.4540 | 19.0669 | 17.7428 | 18.0706 | 15.9987 | 84.42 |
| MoGe-B | 16.1116 | 16.5603 | 15.8885 | 15.5833 | 14.3758 | 92.21 |
| MoGe-L | 13.7934 | 13.6061 | 13.3924 | 13.4427 | 11.9216 | 146.58 |

Errors are angular degrees; lower is better. GSO uses93 validation objects,4 views and5 pitches (1,860 images). Real controls contain10 DiLiGenT and14 LUCES objects. Ten-light panels contain100 and140 independent monocular inputs, averaging errors within each object then objects equally. This does not denote joint multi-image inference.

D2 is BF16; MoGe is FP16. The matched3090 timing protocol and separate full-official/FP32 timings are in the [scale confirmation](../moge2_scale_confirmation_20261106/README.md). Model loading is excluded; PNG read, preprocessing/transfer, prediction and common256-normal PNG write are included. D2 accuracy averages the errors of8 independent inference draws; this is not an8-vector ensemble whose latency would be larger.

MoGe S/B/L have35,103,656 /104,743,592 /330,901,544 parameters. All6,372 MoGe metric rows and264 fresh D2 real-data rows passed exact frozen input-ID sets, image/object/pitch bindings, registration signatures and independent macro aggregation checks. No failed image was omitted. [validation.json](validation.json) contains aggregate counts only.

MoGe-L GSO is13.793366deg in this rerun versus13.793539deg in the previous full-inference snapshot. Source and precision/backend settings are retained. See [protocol.json](protocol.json) and the [data/license audit](../moge2_scale_confirmation_20261106/data_and_license.json). GSO-test remains sealed. The deferred formal3-training-seed comparison is not claimed complete.
