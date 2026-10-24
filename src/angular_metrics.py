"""Shared original angular summary, without historical face-data dependencies."""
import numpy as np
def metrics_from_angles(angles):
    if not len(angles) or not np.isfinite(angles).all():
        raise RuntimeError("Invalid evaluation angles")
    return {"mean": float(angles.mean()), "median": float(np.median(angles)),
            "rmse": float(np.sqrt(np.mean(angles ** 2))),
            "lt_11_25": float(np.mean(angles < 11.25) * 100),
            "lt_20": float(np.mean(angles < 20) * 100),
            "lt_22_5": float(np.mean(angles < 22.5) * 100),
            "lt_25": float(np.mean(angles < 25) * 100),
            "lt_30": float(np.mean(angles < 30) * 100)}
