"""Masked FM/normal losses; unit penalty uses the unnormalized endpoint."""
import torch
import torch.nn.functional as F


def geometry_loss(prediction, target, valid, *, mse_weight=1.0,
                  unit_weight=0.25, cosine_weight=0.25):
    if valid.ndim == 4:
        valid = valid[:, 0]
    valid = valid.bool()
    count = int(valid.sum())
    if not count:
        return None, {"valid_pixels": 0, "skip_batch": True}
    expanded = valid[:, None].expand_as(prediction)
    if not torch.isfinite(prediction[expanded]).all() or not torch.isfinite(target[expanded]).all():
        raise ValueError("Non-finite normal on a valid pixel")
    pred = torch.where(expanded, prediction.float(), 0)
    gt = torch.where(expanded, target.float(), 0)
    mse = (pred - gt).square().sum() / (3 * count)
    unit = ((pred.norm(dim=1) - 1).square())[valid].mean()
    cosine = (1 - F.cosine_similarity(pred, gt, dim=1, eps=1e-8))[valid].mean()
    total = mse_weight * mse + unit_weight * unit + cosine_weight * cosine
    return total, {"valid_pixels": count, "skip_batch": False,
                   "mse": float(mse.detach()), "unit": float(unit.detach()),
                   "cosine": float(cosine.detach())}


def velocity_loss(pred_v, target_v, z_t, normal_gt, times, valid,
                  unit_weight=0.25, cosine_weight=0.25):
    mask = valid[:, 0] if valid.ndim == 4 else valid
    mask = mask.bool()
    count = int(mask.sum())
    if not count:
        return None, {"valid_pixels": 0, "skip_batch": True}
    expanded = mask[:, None].expand_as(pred_v)
    if not torch.isfinite(pred_v[expanded]).all() or not torch.isfinite(target_v[expanded]).all():
        raise ValueError("Non-finite velocity on a valid pixel")
    pv = torch.where(expanded, pred_v.float(), 0)
    tv = torch.where(expanded, target_v.float(), 0)
    mse = (pv - tv).square().sum() / (3 * count)
    endpoint = z_t.float() + (1 - times.view(-1, 1, 1, 1)) * pred_v.float()
    geometric, info = geometry_loss(endpoint, normal_gt, mask, mse_weight=0,
                                    unit_weight=unit_weight, cosine_weight=cosine_weight)
    info["velocity_mse"] = float(mse.detach())
    return mse + geometric, info
