"""Cache historical data once, retaining the exact historical face-mask policy."""
import random
from pathlib import Path
import numpy as np
import cv2
import dlib
from PIL import Image
from scripts.legacy import eval_facescape_modes as legacy
from src.research_io import BEST


def select_pairs(subjects, seed):
    all_pairs = legacy.collect_pairs(str(BEST / "datasets/FacescapeRender1"))
    if not all_pairs:
        raise RuntimeError("No historical pairs found")
    if subjects == 0:
        return all_pairs, len(all_pairs)
    grouped = {}
    for pair in all_pairs:
        grouped.setdefault(pair["subject"], []).append(pair)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(sorted(grouped), min(subjects, len(grouped))))
    return [rng.choice(grouped[subject]) for subject in chosen], len(all_pairs)


def setup_face_mask():
    predictor = BEST / "data/shape_predictor_68_face_landmarks.dat"
    if not predictor.is_file():
        raise FileNotFoundError(predictor)
    legacy._dlib_detector = dlib.get_frontal_face_detector()
    legacy._dlib_predictor = dlib.shape_predictor(str(predictor))


def cache_pairs(pairs, size, device):
    setup_face_mask()
    preprocess = legacy.get_preprocess(size)
    records = []
    for index, pair in enumerate(pairs):
        rgb = Image.open(pair["rgb_path"]).convert("RGB")
        gt = np.asarray(Image.open(pair["normal_path"]).convert("RGB").resize(
            (size, size), Image.BICUBIC)).astype(np.float32) / 255
        face_mask = legacy.create_face_mask(rgb, size)
        foreground = (gt.sum(axis=-1) > .1).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        mask = cv2.erode(foreground,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
        if face_mask is not None:
            mask &= face_mask
        if not mask.any():
            raise RuntimeError(f"No valid historical mask: {pair['rgb_path']}")
        gt_vector = gt.astype(np.float64) * 2 - 1
        gt_normal = gt_vector / (np.linalg.norm(gt_vector, axis=-1, keepdims=True) + 1e-8)
        records.append({"id": f"{index:04d}_{pair['subject']}_{Path(pair['rgb_path']).stem}",
                        "pair": pair, "rgb": rgb, "condition": preprocess(rgb).unsqueeze(0).to(device),
                        "gt_encoded": gt, "gt_normal": gt_normal, "mask": mask,
                        "mask_fallback": face_mask is None})
    return records


def angle_map(normal, record):
    prediction = normal.astype(np.float64)
    cosine = np.clip((prediction * record["gt_normal"]).sum(axis=-1), -1, 1)
    return np.degrees(np.arccos(cosine))


from src.angular_metrics import metrics_from_angles


def prediction_metrics(raw, normal, record):
    # Historical path: channel clamp then normalization, without a PNG round trip.
    encoded = (np.clip(raw, -1, 1) + 1) / 2
    legacy_vector = encoded.astype(np.float64) * 2 - 1
    legacy_normal = legacy_vector / (np.linalg.norm(legacy_vector, axis=-1, keepdims=True) + 1e-8)
    return {"legacy": metrics_from_angles(angle_map(legacy_normal, record)[record["mask"]]),
            "float": metrics_from_angles(angle_map(normal, record)[record["mask"]])}
