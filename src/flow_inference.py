"""Seed-explicit Euler inference, with untouched historical model architecture."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F


@torch.inference_mode()
def infer(model, condition, *, steps=1, mode="geom_cfg", seed=42,
          scale=1.5, amp=True, initial_noise=None, stats_mask=None):
    if steps < 1:
        raise ValueError("steps must be positive")
    if mode not in {"no_cfg", "fixed_cfg", "geom_cfg"}:
        raise ValueError(mode)
    generator = torch.Generator(device=condition.device).manual_seed(int(seed))
    z = torch.randn(condition.shape, generator=generator,
                    device=condition.device, dtype=condition.dtype)
    if initial_noise is not None:
        if initial_noise.shape != condition.shape:
            raise ValueError("Initial state shape mismatch")
        z = initial_noise.clone()
    zero_condition = torch.zeros_like(condition)
    dt = 1.0 / steps
    trace = []
    for t in torch.arange(steps, device=z.device, dtype=torch.float32) / steps:
        times = t.expand(z.shape[0])
        context = torch.cuda.amp.autocast(enabled=amp) if z.is_cuda else nullcontext()
        with context:
            v_c = model(z, times, condition)
            v_u = model(z, times, zero_condition) if mode != "no_cfg" else None
        if mode == "geom_cfg":
            t_view = times.view(-1, 1, 1, 1)
            n_c = F.normalize(z + v_c * (1 - t_view), dim=1)
            n_u = F.normalize(z + v_u * (1 - t_view), dim=1)
            diff = 1.0 - F.cosine_similarity(n_c, n_u, dim=1)
            guidance = (scale * (1.0 - diff)).clamp(0.0, scale).unsqueeze(1)
            # Retain the historical mixed-precision subtraction order for parity.
            velocity = v_u + guidance * (v_c - v_u)
            values = guidance[:, 0]
            if stats_mask is not None:
                values = values[stats_mask.bool()]
            else:
                values = values.flatten()
            if not values.numel():
                raise ValueError("Empty guidance statistics region")
            trace.append({"t": float(t), "mean": float(values.mean()),
                          "zero_fraction": float((values == 0).float().mean()),
                          "lt_0_1_fraction": float((values < 0.1).float().mean()),
                          "gt_1_fraction": float((values > 1).float().mean()),
                          "region": "evaluation_mask" if stats_mask is not None else "whole_image"})
        elif mode == "fixed_cfg":
            velocity = v_u + scale * (v_c - v_u)
        else:
            velocity = v_c
        z = z + velocity * dt
    raw = z.float()
    if not torch.isfinite(raw).all():
        raise ValueError("Non-finite model prediction")
    norm = raw.norm(dim=1, keepdim=True)
    invalid = norm < 1e-8
    normal = F.normalize(raw, dim=1, eps=1e-8)
    return {"raw": raw, "normal": normal, "invalid": invalid,
            "trace": trace, "nfe": steps * (1 if mode == "no_cfg" else 2)}
