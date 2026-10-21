"""Explicit mask and geometric utilities for subsequent general training."""
import numpy as np
import cv2


def resize_normals(normal, valid, size_wh):
    finite = np.isfinite(normal).all(axis=-1)
    valid = np.asarray(valid, dtype=bool) & finite
    clean = np.where(valid[..., None], normal, 0).astype(np.float32)
    weight = cv2.resize(valid.astype(np.float32), size_wh, interpolation=cv2.INTER_LINEAR)
    vectors = cv2.resize(clean, size_wh, interpolation=cv2.INTER_LINEAR)
    vectors = vectors / np.maximum(weight[..., None], 1e-8)
    nearest = cv2.resize(valid.astype(np.uint8), size_wh,
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    lengths = np.linalg.norm(vectors, axis=-1)
    # Require full valid support to avoid introducing invalid-boundary mixtures.
    output_valid = nearest & (weight >= 1 - 1e-6) & (lengths > 1e-8)
    normalized = vectors / np.maximum(lengths[..., None], 1e-8)
    return np.where(output_valid[..., None], normalized, 0).astype(np.float32), output_valid


def crop_resize_intrinsics(intrinsics, crop_xywh, output_wh):
    """Continuous image-plane affine; caller must use matching pixel-center convention."""
    x, y, width, height = crop_xywh
    out_width, out_height = output_wh
    if min(width, height, out_width, out_height) <= 0:
        raise ValueError("Invalid crop/resize size")
    sx, sy = out_width / width, out_height / height
    transform = np.array([[sx, 0, -sx * x], [0, sy, -sy * y], [0, 0, 1]])
    return transform @ np.asarray(intrinsics)


def mirror_normals(normal, valid):
    result = np.flip(normal, axis=1).copy()
    result[..., 0] *= -1
    return result, np.flip(valid, axis=1).copy()
