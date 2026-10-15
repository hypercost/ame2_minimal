"""Synthetic training data and input augmentation for the per-frame mapping model.

Paper Sec. V-B2 lists the augmentations applied to sampled local grids:
additive uniform noise with random magnitude; random cropping from the four
borders; simulated occlusions with random sensor poses and field of view;
random clipping ranges; random missing points and outliers. Exact ranges are
not published -- every number below is our own choice (see ``AugmentCfg``).

Two data sources are supported by the training script:

* ``synthetic_terrain_batch`` -- procedurally generated 2.5D terrains (steps,
  stairs, boxes, pits, stepping stones, slopes, rough noise) rendered
  directly on the local grid, i.e. no simulator needed. The input is produced
  from the ground truth with :func:`simulate_occlusion` + :func:`augment_inputs`.
* data recorded in Isaac Lab with ``ame2/scripts/mapping/collect_mapping_data.py``
  (real Mid360 ray pattern on the locomotion terrains) -- only
  :func:`augment_inputs` is applied on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .grid import LocalGridSpec


@dataclass
class AugmentCfg:
    noise_max: float = 0.05            # per-sample uniform noise half-width ~ U(0, noise_max)
    crop_max_frac: float = 0.2         # per-border crop ~ U(0, crop_max_frac) of the size, prob crop_prob
    crop_prob: float = 0.3
    clip_high_range: tuple[float, float] = (-0.2, 2.0)   # sensor-relative upper clip ~ U(range)
    clip_prob: float = 0.3
    missing_max: float = 0.3           # i.i.d. missing ratio ~ U(0, missing_max)
    outlier_max: float = 0.02          # outlier ratio ~ U(0, outlier_max)
    outlier_range: tuple[float, float] = (-2.5, 0.5)
    range_dropout_max: float = 0.25    # extra missing prob per metre of horizontal range ~ U(0, .)
    # occlusion (synthetic data only)
    sensor_xy_jitter: float = 0.1
    sensor_z_jitter: float = 0.05
    fov_min_deg: float = 120.0         # horizontal FOV ~ U(fov_min, 360)
    fov_prob: float = 0.5              # prob. of restricting the horizontal FOV at all
    blind_radius_max: float = 0.6      # self-occlusion / blind zone radius ~ U(0, .)
    occlusion_samples: int = 48


# -----------------------------------------------------------------------------
# synthetic terrains
# -----------------------------------------------------------------------------


def _rand(n, lo, hi, device, g=None):
    return lo + (hi - lo) * torch.rand(n, device=device, generator=g)


def synthetic_terrain_batch(
    batch: int,
    spec: LocalGridSpec,
    device: torch.device | str = "cpu",
    sensor_height_range: tuple[float, float] = (0.9, 1.5),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random 2.5D terrains on the local grid, heights relative to the sensor.

    Returns (B, L, W) ground-truth elevations clamped to ``[hole_value, z_max]``.
    """
    g = generator
    X, Y = spec.axes(device)
    L, W = X.shape
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)
    B = batch

    def r(lo, hi):
        return _rand(B, lo, hi, device, g).view(B, 1, 1)

    # random in-plane rotation per sample for oriented features
    th = r(-math.pi, math.pi)
    U = torch.cos(th) * X + torch.sin(th) * Y
    V = -torch.sin(th) * X + torch.cos(th) * Y

    # 0 flat/slope, 1 stairs, 2 boxes, 3 pits/gaps, 4 stepping stones, 5 rough
    kind = torch.randint(0, 6, (B,), device=device, generator=g).view(B, 1, 1)

    slope = torch.tan(r(-0.35, 0.35)) * U
    z = torch.zeros(B, L, W, device=device)

    # stairs along U
    step_w = r(0.15, 0.45)
    step_h = r(-0.25, 0.25)
    stairs = torch.floor((U - r(-1.0, 1.0)) / step_w).clamp(-12, 12) * step_h

    # random boxes (stacked / overlapping)
    nbox = 12
    bx = _rand(B * nbox, -3.0, 3.0, device, g).view(B, nbox, 1, 1)
    by = _rand(B * nbox, -1.5, 1.5, device, g).view(B, nbox, 1, 1)
    bw = _rand(B * nbox, 0.1, 1.2, device, g).view(B, nbox, 1, 1)
    bl = _rand(B * nbox, 0.1, 1.2, device, g).view(B, nbox, 1, 1)
    bh = _rand(B * nbox, 0.05, 0.8, device, g).view(B, nbox, 1, 1)
    inside = ((U.unsqueeze(1) - bx).abs() < bl / 2) & ((V.unsqueeze(1) - by).abs() < bw / 2)
    boxes = (inside.float() * bh).sum(dim=1)  # stacked boxes add up

    # pits / gaps: negative boxes
    pits = -(inside[:, : nbox // 2].float() * (bh[:, : nbox // 2] * 2.0)).amax(dim=1)

    # stepping stones: pillars on a jittered lattice above a deep floor
    pitch = r(0.35, 0.8)
    size = pitch * r(0.4, 0.85)
    su = torch.remainder(U - r(0, 1), pitch) < size
    sv = torch.remainder(V - r(0, 1), pitch) < size
    stone_h = r(-0.1, 0.2)
    stones = torch.where(su & sv, stone_h.expand(B, L, W), -r(0.5, 1.5).expand(B, L, W))
    # keep the area right under the robot walkable
    under = (X.abs() < 0.35) & (Y.abs() < 0.35)
    stones = torch.where(under, torch.zeros_like(stones), stones)

    # rough: low-res noise upsampled
    lo_res = torch.randn(B, 1, 9, 5, device=device, generator=g) * r(0.02, 0.25).view(B, 1, 1, 1)
    rough = F.interpolate(lo_res, size=(L, W), mode="bilinear", align_corners=True)[:, 0]
    fine = 0.02 * torch.rand(B, L, W, device=device, generator=g)

    z = torch.where(kind == 1, stairs, z)
    z = torch.where(kind == 2, boxes, z)
    z = torch.where(kind == 3, pits, z)
    z = torch.where(kind == 4, stones, z)
    z = torch.where(kind == 5, rough, z)
    z = z + slope * (kind != 4).float() + fine
    # keep a feasible support patch roughly under the robot for non-flat kinds
    support = (X.abs() < 0.3) & (Y.abs() < 0.3)
    z = torch.where(support & (kind == 3), torch.zeros_like(z), z)

    z = z - r(*sensor_height_range)
    return z.clamp(spec.hole_value, spec.z_max)


# -----------------------------------------------------------------------------
# occlusion + corruptions
# -----------------------------------------------------------------------------


def simulate_occlusion(
    gt: torch.Tensor,
    spec: LocalGridSpec,
    cfg: AugmentCfg = AugmentCfg(),
    generator: torch.Generator | None = None,
    tol: float = 0.02,
) -> torch.Tensor:
    """2.5D line-of-sight test from a randomly jittered sensor near the grid centre.

    A cell is visible if the straight segment from the sensor to the cell's top
    surface never passes below the terrain (sampled at ``occlusion_samples``
    points). Also applies a random horizontal FOV sector and a blind radius.

    Args:
        gt: (B, L, W) sensor-relative ground truth.

    Returns:
        (B, L, W) bool visibility mask.
    """
    g = generator
    B, L, W = gt.shape
    device = gt.device
    X, Y = spec.axes(device)
    sx = _rand(B, -cfg.sensor_xy_jitter, cfg.sensor_xy_jitter, device, g).view(B, 1, 1)
    sy = _rand(B, -cfg.sensor_xy_jitter, cfg.sensor_xy_jitter, device, g).view(B, 1, 1)
    sz = _rand(B, -cfg.sensor_z_jitter, cfg.sensor_z_jitter, device, g).view(B, 1, 1)

    K = cfg.occlusion_samples
    ts = torch.linspace(0.02, 0.98, K, device=device)
    Xv, Yv = X.view(1, 1, L, W), Y.view(1, 1, L, W)
    sx4, sy4, sz4 = sx.unsqueeze(1), sy.unsqueeze(1), sz.unsqueeze(1)
    gt4 = gt.unsqueeze(1)
    visible = torch.ones(B, L, W, dtype=torch.bool, device=device)
    dist_xy = torch.sqrt((Xv - sx4) ** 2 + (Yv - sy4) ** 2)  # (B,1,L,W) sensor -> cell
    chunk = 8  # bound memory: B x chunk x L x W per step
    for k0 in range(0, K, chunk):
        t = ts[k0 : k0 + chunk].view(1, -1, 1, 1)
        kc = t.shape[1]
        px = sx4 + t * (Xv - sx4)  # (B,kc,L,W)
        py = sy4 + t * (Yv - sy4)
        pz = sz4 + t * (gt4 - sz4)
        # normalised sampling coords (grid_sample: x -> width (Y), y -> height (X))
        gx = 2.0 * (py - spec.y_range[0]) / (spec.y_range[1] - spec.y_range[0]) - 1.0
        gy = 2.0 * (px - spec.x_range[0]) / (spec.x_range[1] - spec.x_range[0]) - 1.0
        grid = torch.stack([gx.expand(B, kc, L, W), gy.expand(B, kc, L, W)], dim=-1).view(B, kc * L, W, 2)
        terr = F.grid_sample(gt4, grid, mode="nearest", padding_mode="border", align_corners=True)
        # ignore samples inside the target cell itself (nearest sampling would self-occlude)
        near_target = ((1.0 - t) * dist_xy) < 0.75 * spec.res
        blocked = (terr.view(B, kc, L, W) > pz + tol) & ~near_target
        visible &= ~blocked.any(dim=1)

    # horizontal FOV sector
    az = torch.atan2(Y.unsqueeze(0) - sy, X.unsqueeze(0) - sx)
    heading = _rand(B, -math.pi, math.pi, device, g).view(B, 1, 1)
    fov = torch.deg2rad(_rand(B, cfg.fov_min_deg, 360.0, device, g)).view(B, 1, 1)
    use_fov = (torch.rand(B, device=device, generator=g) < cfg.fov_prob).view(B, 1, 1)
    dang = torch.remainder(az - heading + math.pi, 2 * math.pi) - math.pi
    in_fov = (dang.abs() <= fov / 2) | ~use_fov
    # blind radius
    rad = torch.sqrt((X.unsqueeze(0) - sx) ** 2 + (Y.unsqueeze(0) - sy) ** 2)
    blind = rad < _rand(B, 0.0, cfg.blind_radius_max, device, g).view(B, 1, 1)
    return visible & in_fov & ~blind


def augment_inputs(
    x: torch.Tensor,
    spec: LocalGridSpec,
    cfg: AugmentCfg = AugmentCfg(),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Noise, border crop, clipping, range dropout, missing points and outliers.

    Args:
        x: (B, L, W) model-input grid (holes already at ``spec.hole_value``).
    """
    g = generator
    B, L, W = x.shape
    device = x.device
    hole = spec.hole_value
    valid = x > hole + 0.1
    X, Y = spec.axes(device)

    # additive uniform noise with random magnitude
    mag = _rand(B, 0.0, cfg.noise_max, device, g).view(B, 1, 1)
    x = x + (2 * torch.rand(x.shape, device=device, generator=g) - 1) * mag

    # random clipping range (upper)
    hi = _rand(B, *cfg.clip_high_range, device, g).view(B, 1, 1)
    do_clip = (torch.rand(B, device=device, generator=g) < cfg.clip_prob).view(B, 1, 1)
    x = torch.where(do_clip, torch.minimum(x, hi), x)

    # range-dependent dropout (sparser returns far away)
    rng = torch.sqrt(X**2 + Y**2).unsqueeze(0)
    p_rng = _rand(B, 0.0, cfg.range_dropout_max, device, g).view(B, 1, 1) * rng
    drop = torch.rand(x.shape, device=device, generator=g) < p_rng
    # i.i.d. missing points
    p_miss = _rand(B, 0.0, cfg.missing_max, device, g).view(B, 1, 1)
    drop |= torch.rand(x.shape, device=device, generator=g) < p_miss
    valid = valid & ~drop

    # border crop
    do_crop = (torch.rand(B, device=device, generator=g) < cfg.crop_prob).view(B, 1, 1)
    ii = torch.arange(L, device=device).view(1, L, 1)
    jj = torch.arange(W, device=device).view(1, 1, W)
    c_top = (_rand(B, 0, cfg.crop_max_frac, device, g) * L).long().view(B, 1, 1)
    c_bot = (_rand(B, 0, cfg.crop_max_frac, device, g) * L).long().view(B, 1, 1)
    c_lft = (_rand(B, 0, cfg.crop_max_frac, device, g) * W).long().view(B, 1, 1)
    c_rgt = (_rand(B, 0, cfg.crop_max_frac, device, g) * W).long().view(B, 1, 1)
    keep = (ii >= c_top) & (ii < L - c_bot) & (jj >= c_lft) & (jj < W - c_rgt)
    valid = valid & (keep | ~do_crop)

    # outliers (only where something is reported)
    p_out = _rand(B, 0.0, cfg.outlier_max, device, g).view(B, 1, 1)
    outl = torch.rand(x.shape, device=device, generator=g) < p_out
    lo, hi_o = cfg.outlier_range
    out_val = lo + (hi_o - lo) * torch.rand(x.shape, device=device, generator=g)
    x = torch.where(outl, out_val, x)
    valid = valid | outl

    x = torch.where(valid, x.clamp(hole, spec.z_max), torch.full_like(x, hole))
    return x


def make_synthetic_batch(
    batch: int,
    spec: LocalGridSpec,
    cfg: AugmentCfg = AugmentCfg(),
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    clean: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(input, target, label_mask), each (B, 1, L, W).

    ``clean=True`` gives the guide's "noise free and naive sampling" stage:
    occlusion only, no corruptions.
    """
    gt = synthetic_terrain_batch(batch, spec, device=device, generator=generator)
    vis = simulate_occlusion(gt, spec, cfg, generator=generator)
    x = torch.where(vis, gt, torch.full_like(gt, spec.hole_value))
    if not clean:
        x = augment_inputs(x, spec, cfg, generator=generator)
    mask = torch.ones_like(gt, dtype=torch.bool)
    return x.unsqueeze(1), gt.unsqueeze(1), mask.unsqueeze(1)
