"""Simple depth/lidar point-cloud corruption for student training.

Paper (Appendix B): "During student training, we make 15% of depth cloud
points missing and 2% artifacts." The artifact model is not specified; we
pull a hit towards the sensor along its ray by a random fraction (a crude
mixed-pixel / spurious-return model) and add small isotropic jitter.
"""

from __future__ import annotations

import torch


def apply_pointcloud_noise(
    hits_w: torch.Tensor,
    origin_w: torch.Tensor,
    drop_ratio: float = 0.15,
    artifact_ratio: float = 0.02,
    artifact_range_frac: tuple[float, float] = (0.2, 0.95),
    jitter_std: float = 0.0,
) -> torch.Tensor:
    """Return a corrupted copy of ``hits_w``.

    Args:
        hits_w: (n, R, 3) world hit points; misses are non-finite and stay misses.
        origin_w: (n, 3) sensor origins.
        drop_ratio: fraction of points turned into misses (``inf``).
        artifact_ratio: fraction of points moved along their ray to a random
            fraction of the measured range.
        artifact_range_frac: range of that fraction.
        jitter_std: optional Gaussian jitter (m) on all valid points.
    """
    out = hits_w.clone()
    finite = torch.isfinite(out).all(dim=-1)  # (n, R)
    if jitter_std > 0:
        out = torch.where(finite.unsqueeze(-1), out + jitter_std * torch.randn_like(out), out)
    if artifact_ratio > 0:
        art = finite & (torch.rand(finite.shape, device=out.device) < artifact_ratio)
        lo, hi = artifact_range_frac
        frac = lo + (hi - lo) * torch.rand(finite.shape, device=out.device)
        o = origin_w.unsqueeze(1)
        moved = o + (out - o) * frac.unsqueeze(-1)
        out = torch.where(art.unsqueeze(-1), moved, out)
    if drop_ratio > 0:
        drop = torch.rand(finite.shape, device=out.device) < drop_ratio
        out = torch.where(drop.unsqueeze(-1), torch.full_like(out, float("inf")), out)
    return out
