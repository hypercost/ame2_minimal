"""Terrain meshes for mapping-model data sampling (paper Sec. V-B1, Fig. 6).

Four families, each built as a plain ``(vertices (V,3) float32, faces (F,3) int32)``
numpy mesh, no Isaac / Kit needed:

* ``"locomotion"``  -- the locomotion training terrains. These come from Isaac Lab's
  ``TerrainGenerator`` (``AME2_TERRAINS_CFG_G1``). ``isaaclab.terrains`` imports
  ``isaaclab.sim`` (omni/pxr) at package import, so it cannot run without Kit:
  export once with ``ame2/scripts/mapping/export_terrain_mesh.py`` and load the
  ``.npz`` with :func:`load_terrain_npz`.
* ``"stacked_boxes"`` -- random boxes on a plane, partly stacked on each other.
* ``"heightfield"``   -- random height fields (multi-scale noise, random slopes,
  random step quantisation) per tile.
* ``"floating_boxes"`` -- boxes hovering above a (slightly rough) ground plane.

All sizes/ranges are our own choice; the paper only names the families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAMILIES = ("locomotion", "stacked_boxes", "heightfield", "floating_boxes")


@dataclass
class TerrainMesh:
    name: str
    vertices: np.ndarray        # (V, 3) float32
    faces: np.ndarray           # (F, 3) int32
    bounds: tuple[float, float, float, float]  # x0, x1, y0, y1 of the walkable area
    probe_z: float              # start height of the downward "where does the robot stand" probe
    # optional spawn points (e.g. terrain-generator origins); poses are sampled around them
    origins: np.ndarray | None = None


# ----------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------

_BOX_FACES = np.array(
    [[0, 2, 1], [0, 3, 2],  # bottom
     [4, 5, 6], [4, 6, 7],  # top
     [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
     [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
    dtype=np.int32,
)


def boxes_mesh(center_xy: np.ndarray, size_xy: np.ndarray, z0: np.ndarray, z1: np.ndarray, yaw: np.ndarray):
    """N oriented boxes (vectorised). Returns (vertices, faces)."""
    n = center_xy.shape[0]
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    sx, sy = 0.5 * size_xy[:, 0:1], 0.5 * size_xy[:, 1:2]
    cx = np.array([-1, 1, 1, -1], np.float32)
    cy = np.array([-1, -1, 1, 1], np.float32)
    lx, ly = cx[None] * sx, cy[None] * sy                     # (n, 4)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    wx = center_xy[:, 0:1] + c * lx - s * ly
    wy = center_xy[:, 1:2] + s * lx + c * ly
    bot = np.stack([wx, wy, np.repeat(z0[:, None], 4, 1)], -1)
    top = np.stack([wx, wy, np.repeat(z1[:, None], 4, 1)], -1)
    v = np.concatenate([bot, top], 1).reshape(-1, 3).astype(np.float32)
    f = (_BOX_FACES[None] + 8 * np.arange(n, dtype=np.int32)[:, None, None]).reshape(-1, 3)
    return v, f


def plane_mesh(x0, x1, y0, y1, z=0.0):
    v = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    return v, f


def heightfield_mesh(h: np.ndarray, res: float, x0: float = 0.0, y0: float = 0.0):
    """Regular-grid height field (L, W) -> triangle mesh (vertex at each sample)."""
    L, W = h.shape
    xs = x0 + res * np.arange(L, dtype=np.float32)
    ys = y0 + res * np.arange(W, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    v = np.stack([X, Y, h.astype(np.float32)], -1).reshape(-1, 3)
    i = np.arange(L - 1)[:, None] * W + np.arange(W - 1)[None]
    i = i.reshape(-1).astype(np.int32)
    f = np.concatenate([np.stack([i, i + W, i + W + 1], 1), np.stack([i, i + W + 1, i + 1], 1)], 0)
    return v, f.astype(np.int32)


def merge(*meshes):
    vs, fs, off = [], [], 0
    for v, f in meshes:
        vs.append(v)
        fs.append(f + off)
        off += v.shape[0]
    return np.concatenate(vs).astype(np.float32), np.concatenate(fs).astype(np.int32)


# ----------------------------------------------------------------------------
# procedural families
# ----------------------------------------------------------------------------

def stacked_boxes(rng: np.random.Generator, size: float = 40.0, n_base: int = 900, layers: int = 3) -> TerrainMesh:
    meshes = [plane_mesh(-5, size + 5, -5, size + 5, 0.0)]
    c = rng.uniform(0, size, (n_base, 2))
    s = rng.uniform(0.2, 2.0, (n_base, 2))
    z1 = rng.uniform(0.03, 0.5, n_base)
    yaw = rng.uniform(-np.pi, np.pi, n_base)
    meshes.append(boxes_mesh(c, s, np.zeros(n_base), z1, yaw))
    for _ in range(layers):  # stack smaller boxes on random existing ones
        k = rng.integers(0, c.shape[0], c.shape[0] // 2)
        off = rng.uniform(-0.4, 0.4, (k.size, 2)) * s[k]
        s2 = s[k] * rng.uniform(0.3, 0.9, (k.size, 2))
        z0 = z1[k]
        z2 = z0 + rng.uniform(0.03, 0.4, k.size)
        y2 = yaw[k] + rng.uniform(-0.5, 0.5, k.size)
        meshes.append(boxes_mesh(c[k] + off, s2, z0, z2, y2))
        c, s, z1, yaw = c[k] + off, s2, z2, y2
    v, f = merge(*meshes)
    return TerrainMesh("stacked_boxes", v, f, (0.0, size, 0.0, size), probe_z=10.0)


def _smooth_noise(rng, L, W, cell):
    """Bilinear-upsampled uniform noise with feature size ``cell`` samples."""
    gl, gw = L // cell + 2, W // cell + 2
    g = rng.uniform(-1, 1, (gl, gw))
    xi = np.arange(L) / cell
    yi = np.arange(W) / cell
    x0, y0 = np.floor(xi).astype(int), np.floor(yi).astype(int)
    fx, fy = (xi - x0)[:, None], (yi - y0)[None]
    a = g[x0][:, y0]
    b = g[x0 + 1][:, y0]
    c = g[x0][:, y0 + 1]
    d = g[x0 + 1][:, y0 + 1]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def random_heightfield(rng: np.random.Generator, size: float = 40.0, res: float = 0.04, tile: float = 5.0) -> TerrainMesh:
    n = int(round(size / res)) + 1
    tn = int(round(tile / res))
    h = np.zeros((n, n), np.float32)
    for i in range(0, n, tn):
        for j in range(0, n, tn):
            L, W = min(tn, n - i), min(tn, n - j)
            amp = rng.uniform(0.0, 0.6)
            t = np.zeros((L, W))
            for cell, w in ((rng.integers(20, 60), 1.0), (rng.integers(4, 15), 0.3), (1, rng.uniform(0, 0.05))):
                t += w * _smooth_noise(rng, L, W, int(cell))
            t *= amp
            gx, gy = rng.uniform(-0.4, 0.4, 2)  # random slope
            t += gx * res * np.arange(L)[:, None] + gy * res * np.arange(W)[None]
            if rng.random() < 0.5:  # quantise -> steps / terraces
                q = rng.uniform(0.05, 0.3)
                t = np.round(t / q) * q
            h[i:i + L, j:j + W] = t - t.mean() + rng.uniform(-0.3, 0.3)
    v, f = heightfield_mesh(h, res)
    return TerrainMesh("heightfield", v, f, (0.0, size, 0.0, size), probe_z=float(h.max()) + 1.0)


def floating_boxes(rng: np.random.Generator, size: float = 40.0, n: int = 300) -> TerrainMesh:
    gn = int(size / 0.1) + 1
    ground = 0.03 * _smooth_noise(rng, gn, gn, 5)
    meshes = [heightfield_mesh(ground.astype(np.float32), 0.1)]
    c = rng.uniform(0, size, (n, 2))
    s = rng.uniform(0.2, 3.0, (n, 2))
    z0 = rng.uniform(0.15, 2.5, n)
    z1 = z0 + rng.uniform(0.03, 0.6, n)
    meshes.append(boxes_mesh(c, s, z0, z1, rng.uniform(-np.pi, np.pi, n)))
    # some grounded clutter as well
    m = n // 4
    meshes.append(boxes_mesh(rng.uniform(0, size, (m, 2)), rng.uniform(0.1, 1.0, (m, 2)), np.zeros(m),
                             rng.uniform(0.03, 0.1, m), rng.uniform(-np.pi, np.pi, m)))
    v, f = merge(*meshes)
    # the robot stands on the ground, below the floating boxes
    return TerrainMesh("floating_boxes", v, f, (0.0, size, 0.0, size), probe_z=0.15)


def load_terrain_npz(path: str) -> TerrainMesh:
    """Load a mesh written by ``export_terrain_mesh.py``."""
    d = np.load(path)
    v = d["vertices"].astype(np.float32)
    f = d["faces"].astype(np.int32)
    if "bounds" in d:
        b = tuple(float(x) for x in d["bounds"])
    else:
        b = (float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()), float(v[:, 1].max()))
    origins = d["origins"].reshape(-1, 3).astype(np.float32) if "origins" in d else None
    return TerrainMesh("locomotion", v, f, b, probe_z=float(v[:, 2].max()) + 1.0, origins=origins)


def build_family(name: str, rng: np.random.Generator, **kw) -> TerrainMesh:
    if name == "stacked_boxes":
        return stacked_boxes(rng, **kw)
    if name == "heightfield":
        return random_heightfield(rng, **kw)
    if name == "floating_boxes":
        return floating_boxes(rng, **kw)
    raise ValueError(f"unknown procedural family {name!r} (locomotion meshes come from load_terrain_npz)")
