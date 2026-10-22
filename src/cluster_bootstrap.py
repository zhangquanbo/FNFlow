"""Identity/scene cluster resampling for a macro-image ratio estimand."""
import numpy as np


def cluster_percentile_intervals(values, clusters, *, draws=10000, seed=20261022):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    clusters = list(clusters)
    if values.ndim != 2 or values.shape[0] != len(clusters) or not np.isfinite(values).all():
        raise ValueError("Need finite image-by-metric values and one cluster per image")
    unique = sorted(set(clusters))
    if len(unique) < 2 or draws < 100:
        raise ValueError("Need multiple clusters and at least 100 resamples")
    lookup = {value: i for i, value in enumerate(unique)}
    group = np.array([lookup[value] for value in clusters])
    count = np.bincount(group, minlength=len(unique))
    sampled = np.random.default_rng(seed).integers(0, len(unique), size=(draws, len(unique)))
    denominators = count[sampled].sum(axis=1)
    results = []
    for column in values.T:
        total = np.bincount(group, weights=column, minlength=len(unique))
        estimates = total[sampled].sum(axis=1) / denominators
        lo, hi = np.percentile(estimates, [2.5, 97.5])
        results.append({"estimate": float(column.mean()), "lower_95": float(lo), "upper_95": float(hi)})
    return {"num_images": len(clusters), "num_clusters": len(unique), "resamples": draws,
            "seed": seed, "interval": "95% percentile identity-cluster bootstrap",
            "estimand": "macro-image mean; whole clusters resampled with replacement, ratio of summed errors to summed image counts",
            "scope": "conditional on fixed trained weights, inference replicates and observed development-domain cluster population; not a universal generalization guarantee",
            "metrics": results}
