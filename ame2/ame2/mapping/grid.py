"""Point cloud -> local 2D max-height grid (stage 1 of the AME-2 mapping pipeline).

Paper (Sec. V-A): "For each sensing frame, we project the point cloud onto a
local 2D height grid. If multiple points fall into the same cell, we keep the
maximum z-value [...] and we assign a fixed minimum value to cells with no
points."  Implemented batched with ``torch.Tensor.scatter_reduce_`` (amax).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalGridSpec:
    """Geometry of the per-frame model input grid.

    The grid is centred at the sensor xy, its axes follow the base yaw, and
    heights are relative to the sensor z. Cell centres are
    ``arange(x_range[0], x_range[1] + eps, res)`` (inclusive), i.e. for the
    default G1/Mid360 setup 201 x 101 cells at 3 cm.
    """

    x_range: tuple[float, float] = (-3.0, 3.0)
    y_range: tuple[float, float] = (-1.5, 1.5)
    res: float = 0.03
    hole_value: float = -3.0
    z_max: float = 2.0

    @property
    def shape(self) -> tuple[int, int]:
        L = int(round((self.x_range[1] - self.x_range[0]) / self.res)) + 1
        W = int(round((self.y_range[1] - self.y_range[0]) / self.res)) + 1
        return L, W

    def axes(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Cell-centre meshgrid ``(X, Y)`` of shape (L, W), ``indexing="ij"``."""
        L, W = self.shape
        xs = self.x_range[0] + self.res * torch.arange(L, dtype=torch.float32, device=device)
        ys = self.y_range[0] + self.res * torch.arange(W, dtype=torch.float32, device=device)
        return torch.meshgrid(xs, ys, indexing="ij")


def points_to_local_grid(
    points_w: torch.Tensor,
    origin_w: torch.Tensor,
    yaw: torch.Tensor,
    spec: LocalGridSpec,
    return_valid: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Scatter world-frame points into per-env local max-height grids.

    Args:
        points_w: (n, R, 3) world points; misses may be ``inf``/``nan`` and are ignored.
        origin_w: (n, 3) grid origin (sensor position) in world.
        yaw: (n,) yaw of the grid axes (base yaw).
        spec: grid geometry.
        return_valid: also return the (n, R) mask of points that landed in the grid.

    Returns:
        (n, L, W) heights relative to ``origin_w[:, 2]``; empty cells are
        ``spec.hole_value``; filled cells are clamped to ``[hole_value, z_max]``.
    """
    n, R = points_w.shape[0], points_w.shape[1]
    device = points_w.device
    rel = points_w - origin_w.unsqueeze(1)
    c = torch.cos(yaw).unsqueeze(1)
    s = torch.sin(yaw).unsqueeze(1)
    # world -> yaw frame (rotate by -yaw)
    lx = c * rel[..., 0] + s * rel[..., 1]
    ly = -s * rel[..., 0] + c * rel[..., 1]
    zr = rel[..., 2]

    x0, x1 = spec.x_range
    y0, y1 = spec.y_range
    L, W = spec.shape
    P = L * W
    # half-cell margin so that every point that rounds onto a border cell is kept
    half = 0.5 * spec.res
    valid = (
        (lx >= x0 - half) & (lx < x1 + half) & (ly >= y0 - half) & (ly < y1 + half) & torch.isfinite(zr)
        & torch.isfinite(lx) & torch.isfinite(ly)
    )

    # nan-safe indices (invalid entries are masked out below anyway)
    lx_s = torch.where(valid, lx, torch.zeros_like(lx))
    ly_s = torch.where(valid, ly, torch.zeros_like(ly))
    ix = torch.round((lx_s - x0) / spec.res).long().clamp_(0, L - 1)
    iy = torch.round((ly_s - y0) / spec.res).long().clamp_(0, W - 1)
    benv = torch.arange(n, device=device).unsqueeze(1).expand(n, R)
    gidx = (benv * P + ix * W + iy)[valid]

    grid = torch.full((n * P,), float("-inf"), device=device, dtype=points_w.dtype)
    grid.scatter_reduce_(0, gidx, zr[valid], reduce="amax", include_self=True)
    grid = grid.view(n, L, W)
    filled = torch.isfinite(grid)
    em = torch.where(filled, grid.clamp(spec.hole_value, spec.z_max), torch.full_like(grid, spec.hole_value))
    if return_valid:
        return em, valid
    return em


def fill_isolated_holes(grid: torch.Tensor, hole_value: float = -3.0, min_neighbors: int = 3) -> torch.Tensor:
    """Fill hole cells that have >= ``min_neighbors`` filled 8-neighbours with their max.

    Used for the dense top-down ground-truth grid: a yaw-rotated ray lattice can
    leave single empty cells after nearest-cell rounding. Real occlusion holes are
    large, so this only closes rasterisation pin-holes.

    Args:
        grid: (n, L, W) heights with holes at ``hole_value``.
    """
    x = grid.unsqueeze(1)
    filled = (x > hole_value + 1e-3).float()
    n_filled = torch.nn.functional.conv2d(filled, torch.ones(1, 1, 3, 3, device=grid.device, dtype=grid.dtype), padding=1) - filled
    neigh_max = torch.nn.functional.max_pool2d(
        torch.where(filled > 0, x, torch.full_like(x, float("-inf"))), 3, stride=1, padding=1
    )
    fill = (filled == 0) & (n_filled >= min_neighbors) & torch.isfinite(neigh_max)
    return torch.where(fill, neigh_max, x).squeeze(1)
