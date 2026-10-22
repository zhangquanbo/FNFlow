"""Stable image-specific noise and separately defined sensitivity statistics."""
import hashlib
import json
from pathlib import PurePosixPath
import numpy as np

PROTOCOL = "sha256_image_noise_v1"


def image_noise_seed(base_seed, image_key):
    """Derive a 63-bit seed from a base seed and a dataset-relative RGB path.

    The key must not depend on image order, an absolute machine path, GT or metrics.
    Python's process-randomized hash() is intentionally not used.
    """
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("Base seed must be an integer")
    if not isinstance(image_key, str) or not image_key or "\\" in image_key:
        raise ValueError("Expected a non-empty relative POSIX image key")
    path = PurePosixPath(image_key)
    if path.is_absolute() or ".." in path.parts or str(path) != image_key:
        raise ValueError("Image key must be canonical and dataset-relative")
    payload = json.dumps([PROTOCOL, base_seed, image_key], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def mae_seed_statistics(errors):
    """Rows are images; columns are paired inference replicates, never training runs."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[0] == 0 or errors.shape[1] < 2:
        raise ValueError("Need images x at least two inference replicates")
    if not np.isfinite(errors).all():
        raise ValueError("Non-finite image error")
    dataset_maes = errors.mean(axis=0)
    image_sd = errors.std(axis=1, ddof=1)
    return {"num_images": errors.shape[0], "num_inference_replicates": errors.shape[1],
            "mean_mae_deg": float(dataset_maes.mean()),
            "dataset_mae_sd_ddof1_deg": float(dataset_maes.std(ddof=1)),
            "mean_image_mae_sd_ddof1_deg": float(image_sd.mean()),
            "median_image_mae_sd_ddof1_deg": float(np.median(image_sd)),
            "p95_image_mae_sd_ddof1_deg": float(np.percentile(image_sd, 95)),
            "dataset_maes_deg": dataset_maes.tolist()}


def directional_sensitivity(samples, mask):
    """Macro within-image variation of unit directions, not calibrated uncertainty."""
    samples = np.asarray(samples, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if samples.ndim != 4 or samples.shape[0] < 2 or samples.shape[-1] != 3:
        raise ValueError("Need samples x height x width x three channels")
    if samples.shape[1:3] != mask.shape or not mask.any():
        raise ValueError("Empty or mismatched statistics mask")
    vectors = samples[:, mask, :]
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(lengths < 1e-8):
        raise ValueError("Invalid directions in statistics mask")
    vectors = vectors / lengths
    center = vectors.mean(axis=0)
    resultant = np.clip(np.linalg.norm(center, axis=-1), 0, 1)
    degenerate = resultant < 1e-12
    dispersion = None
    if not degenerate.any():
        direction = center / resultant[:, None]
        dots = np.clip((vectors * direction[None]).sum(axis=-1), -1, 1)
        dispersion = float(np.degrees(np.arccos(dots)).mean())
    return {"mean_pixel_directional_variance_1_minus_R": float((1 - resultant).mean()),
            "mean_pixel_angular_dispersion_deg": dispersion,
            "undefined_mean_direction_pixels": int(degenerate.sum()),
            "valid_pixels": int(mask.sum())}
