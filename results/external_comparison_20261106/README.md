# Completed external measurements, 2026-11-06

All16 scheduled new measurement scopes, D2 inference variants and RTX3090 efficiency measurements have finished. All22,932 private metric records have been reconciled. StableNormal has not passed independent direction checks and is reported separately as diagnostic. Held-out GSO-test remains sealed. These runs do not constitute multiple independent training seeds.

## Registered baseline results

MAE in degrees, lower is better. Real ten-light panels use IDs21..30 as independent monocular inputs, averaging lights within object and then objects equally. The single-light controls use the prior frozen lowest-ID images. Model-specific input defaults, versions and per-image RNG are recorded in [protocol.json](protocol.json); exact reproduction of RoSE paper tables is not claimed.

| Model | GSO-val five-angle | DiLiGenT one light | LUCES one light | DiLiGenT ten lights | LUCES ten lights |
|---|---:|---:|---:|---:|---:|
| D2 | 18.6650 | 18.2309 | 17.7026 | 18.0665 | 17.0940 |
| NeuralLightRig | 20.5024 | 23.1622 | 25.0410 | 29.2719 | 20.0652 |
| MoGe2 | 13.7935 | 13.6068 | 13.3942 | 13.4425 | 11.9216 |
| NiRNE | 21.8307 | 18.3271 | 19.0732 | 17.9896 | 18.3844 |
| Lotus-D | 27.1948 | 24.8677 | 25.2511 | 22.6992 | 16.9447 |
| Lotus-G | 24.9342 | 24.0328 | 22.4572 | 22.0979 | 16.2589 |
| DSINE | 27.6705 | 26.2489 | 25.4471 | 24.4416 | 22.6094 |
| MarigoldNormal | 27.6370 | 25.1796 | 22.5494 | 24.8045 | 17.5067 |

D2 uses training seed1247 with mean inference errors over seeds42..49. External pretrained models use inference seed42, and Neural LightRig uses seed511. The D2 and Neural LightRig GSO-val cells are their previously published frozen recipe-ladder references; they were not rerun in the new16-scope matrix. In particular, the Neural LightRig GSO reference uses its earlier fresh-stream RNG protocol, while new real panels use image-key-derived noise. No TTA is included in the baseline table.

## D2 optional inference variants

The following real scores use one light per object, separately from the ten-light table above. Each constituent normal prediction is normalized before vector averaging. Flip processing includes the registered RGB/noise reflection and prediction reflection.

| Variant | GSO-val | DiLiGenT one light | LUCES one light |
|---|---:|---:|---:|
| single42 | 18.6566 | 18.2655 | 17.6802 |
| flip_TTA42 | 18.3652 | 17.9348 | 17.6614 |
| mean4 | 18.5533 | 18.1433 | 17.5798 |
| flip_TTA_mean4 | 18.2916 | 17.8646 | 17.5454 |

The original8-error-mean D2 reference is18.6650deg on GSO-val; flip_TTA_mean4 is18.2916deg. The exact single42 control reproduces its frozen per-image reference with maximum drift0.0000deg. [tta.json](tta.json) contains the full variant and angle breakdown.

## RTX3090 batch1 efficiency

Sequential measurements use5 warmups and30 timed samples, synchronizing CUDA each iteration. End-to-end scope includes input PNG reading, preprocessing, model computation, common256 normal conversion and output PNG writing; official Neural LightRig additionally writes its five-column composite. Model loading is excluded. D2 uses BF16, UNet256 and frozen DINOv2-B518; Neural LightRig uses its official512 input and75 denoising steps. The two model environments have their recorded PyTorch versions.

| Model | Median seconds | P90 seconds | Unique total parameters | Peak allocated GiB |
|---|---:|---:|---:|---:|
| D2 | 0.078648 | 0.080148 | 130,400,259 | 0.926 |
| NeuralLightRig | 84.804909 | 84.864190 | 2,211,227,520 | 14.269 |

The measured median-latency ratio is1078.3. D2 was repeated solely to correct a forward-hook instrumentation issue; the original records remain preserved. [efficiency.json](efficiency.json) contains aggregate timing statistics and instrumentation details.

## StableNormal diagnostic record

Raw aggregates are retained below and in summary.json; they are excluded from the main performance table because independent direction checks did not close.

| GSO-val | DiLiGenT one light | LUCES one light | DiLiGenT ten lights | LUCES ten lights |
|---:|---:|---:|---:|---:|
| 73.3637 | 73.0146 | 59.2507 | 74.0737 | 63.3525 |

On4 independent off-center analytic-sphere controls, the adapter reproduces the pinned official public Predictor output exactly after its official display conversion: maximum8-bit RGB difference0 on every control. The official output fails one or more registered cardinal-direction probes on2 controls. This rules out an adapter/public-entry output difference at the tested interface; it does not determine whether coordinate convention or shape-prediction behavior causes the benchmark scores. The fixed candidate matrix and all raw benchmark records are preserved. No alternate axes were selected using benchmark ground truth. See [quality_review.json](quality_review.json).

[validation.json](validation.json) records numeric reconciliation. Runtime failures and versioned corrections were retained locally: Marigold CHW-to-HWC layout correction, an early DSINE source-hash startup race, and StableNormal independent-control amendments. No failed images were silently omitted from completed scopes.

The earlier [recipe ladder and C-D3 boundary supplement](../object_recipe_ladder_20261105/README.md) and [six-model real snapshot](../external_real_20261106/README.md) remain separate historical snapshots.

Publication scope: model/pitch/panel aggregates and aggregate control counts only. Image identifiers, per-image and per-fixture records, images, checkpoints and individual timing samples are excluded. Reconciliation covers the private source records.
