# Completed real-domain scopes, 2026-11-06

This is a snapshot of six fully completed real-domain model scopes. The full external comparison is still running; it is not a final ranking. The additional completed MoGe2 GSO-val scope is reported separately below; remaining external scopes continue to run. Held-out GSO-test remains sealed.

MAE in degrees; lower is better. The single-light controls use the same frozen lowest-ID images as the prior recipe snapshot. The ten-light panels use light IDs21..30:100 images/10 DiLiGenT objects and140 images/14 LUCES objects. Every RGB image is processed independently. Light errors are averaged within object, then objects equally.

| Model | DiLiGenT one light | LUCES one light | DiLiGenT ten lights | LUCES ten lights |
|---|---:|---:|---:|---:|
| D2 | 18.2309 | 17.7026 | 18.0665 | 17.0940 |
| DSINE | 26.2489 | 25.4471 | 24.4416 | 22.6094 |
| MoGe2 | 13.6068 | 13.3942 | 13.4425 | 11.9216 |
| Lotus-D | 24.8677 | 25.2511 | 22.6992 | 16.9447 |
| Lotus-G | 24.0328 | 22.4572 | 22.0979 | 16.2589 |
| NiRNE | 18.3271 | 19.0732 | 17.9896 | 18.3844 |

D2 is one training seed1247, with inference errors averaged across seeds42..49. Each external pretrained model uses one registered inference seed42; Lotus-D is deterministic. These are not multiple independent training seeds. No TTA or normal-vector ensemble is used.

Light selection follows [RoSE AppendixC](https://arxiv.org/html/2602.09929v5#A3). The registered model versions, input defaults and per-image RNG are our own execution conditions, so these are not claimed as exact reproductions of the paper tables. In particular, they must not be merged with the earlier Neural LightRig legacy-order five-seed results.

See [summary.json](summary.json), [protocol.json](protocol.json), and [validation.json](validation.json). Reconciliation checks1,584 private metric rows; no image-level records, images, weights or private manifests are exported.

The prior [recipe result page](../object_recipe_ladder_20261105/README.md) now includes the C/D1/D2/D3 boundary supplement.

The archived v1 inference source and adapter correspond to the listed completed models. Other model branches are outside the validated scope of this snapshot.

## Additional completed GSO-val scope

MoGe2 completed the same frozen93-object, four-view, five-angle GSO-val primary panel (1,860 images). Equal-angle object-macro MAE is **13.7935deg**, compared with the frozen D2 reference of18.6650deg. The input manifests and GT/masks are unchanged; model-specific inference settings follow the registered source. This is not held-out test scoring or a completed full comparison.

| Angle | MoGe2 MAE |
|---|---:|
| 0deg | 13.9461 |
| 15deg | 12.7280 |
| 30deg | 13.0472 |
| 45deg | 13.8326 |
| 60deg | 15.4139 |

[gso_moge2.json](gso_moge2.json) includes scope details, source bindings and reconciliation of all1,860 private metric rows. The real-domain validation file above covers its separate1,584 rows.
