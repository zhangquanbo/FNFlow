# Completed WSD convergence diagnostics

Single training seed 1247, condition dropout p=0, 256px center-square data; no DINO, ray input or CFG.

**Diagnostic training is complete. The formal 24-run comparison is not complete and remains locked.**

Each row below uses a separate 2,500-update linear cooldown restored from a stable checkpoint. Two paired inference replicates (42, 43), image-specific noise, macro-image masked MAE in degrees; lower is better.

| Total updates | Method | Hypersim (700) | Arb (594 views / 198 assets) | NYUv2 (795) |
|---:|---|---:|---:|---:|
| 12500 | R | 28.7167 | 28.6905 | 37.6301 |
| 12500 | V | 30.9014 | 32.9562 | 35.7686 |
| 12500 | Xn | 30.6305 | 33.1789 | 35.1245 |
| 22500 | R | 26.1644 | 26.4377 | 36.7133 |
| 22500 | V | 27.3800 | 31.2865 | 33.7997 |
| 22500 | Xn | 27.2041 | 31.4974 | 33.2262 |
| 32500 | R | 25.9023 | 25.6060 | 36.3643 |
| 32500 | V | 25.4081 | 29.8389 | 32.6820 |
| 32500 | Xn | 25.6919 | 29.7763 | 32.0274 |
| 52500 | R | 24.3455 | 24.9713 | 35.1048 |
| 52500 | V | 23.4378 | 27.3166 | 30.5735 |
| 52500 | Xn | 23.2017 | 27.1311 | 30.8801 |

## Final 52,500-update metrics

| Method | Domain | MAE | Boundary MAE | Mean within-image MAE SD | Mean pixel angular dispersion |
|---|---|---:|---:|---:|---:|
| R | hypersim | 24.3455 | 31.8500 | 0.0000 | 0.0000 |
| R | arb | 24.9713 | 29.4638 | 0.0000 | 0.0000 |
| R | nyuv2 | 35.1048 | 41.8663 | 0.0000 | 0.0000 |
| V | hypersim | 23.4378 | 29.9003 | 1.2142 | 4.4655 |
| V | arb | 27.3166 | 31.0458 | 1.6812 | 4.4917 |
| V | nyuv2 | 30.5735 | 39.2441 | 1.3937 | 5.3726 |
| Xn | hypersim | 23.2017 | 29.7381 | 1.1696 | 3.8566 |
| Xn | arb | 27.1311 | 31.1526 | 1.5751 | 3.7667 |
| Xn | nyuv2 | 30.8801 | 39.2778 | 1.3922 | 4.8129 |

R has no inference noise; its within-image noise metrics are zero by construction. Two-replicate statistics are coarse diagnostics, not calibrated uncertainty, training-seed variation or confidence intervals.

## Same-prediction Arb material sensitivity

| Method | Full 198 usable assets / 594 views | Remaining 170 assets / 510 views |
|---|---:|---:|
| R | 24.9713 | 25.0380 |
| V | 27.3166 | 27.2389 |
| Xn | 27.1311 | 27.1020 |

28 flagged usable validation assets are excluded in the sensitivity subset; the 200-asset source plan has 30 flagged assets, including two without usable near-unit-normal labels. Main metrics and budget decisions retain the original full usable set. Metadata flags do not establish actual rendered transparency or causal label mismatch.

## Matched first-2,500-update clipping window

| Method | Clipped updates (%) | Median preclip L2 norm | P95 preclip L2 norm |
|---|---:|---:|---:|
| R | 38.96 | 0.9041 | 1.8434 |
| V | 33.00 | 0.7895 | 3.5435 |
| Xn | 13.32 | 0.5661 | 1.4323 |
| Xv | 97.84 | 4.1239 | 21.0629 |

All four methods use the unchanged clipping threshold 1.0. Xv has only the 2,500-update clipping diagnostic; it is not a fourth completed long-run method.

## Registered budget status

`inconclusive_still_improving`: at least one registered method/domain improves by more than 0.3 degrees between stable 47.5k and 50k or between cooled 32.5k and 52.5k. No new common budget is approved, no automatic extension beyond the registered cap is enabled, and the formal grid remains locked.

All 16 completed source summaries, their 48 domain metrics, material subset counts/means, controlled signatures and actual budget-trigger values were reconciled. Private per-image inputs are not distributed. Training probes and full stable curves are in the JSON files. GSO preparation summaries describe rendering readiness, not model performance.

[Aggregate JSON](summary.json) · [Stable learning curves](stable_learning_curves.json)
