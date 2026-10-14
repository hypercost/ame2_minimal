"""Online Warp ray-tracing sampler for mapping-model training data (paper Sec. V-B1/2).

No physics and no Isaac/Kit: random robot poses are sampled above terrain meshes
(:mod:`ame2.mapping.terrain_meshes`) and two ray sets are traced per frame with
``warp.mesh_query_ray``:

* **label**: a top-down ray per cell of the model grid -> ground-truth elevation
  (sensor-relative, clamped to ``[hole, z_max]`` like the input);
* **input**: rays from a randomly perturbed sensor pose with either the real
  Mid360 pattern (random 20k-ray window of ``sensors/mid360_raydirs.npy``, same
  mount as in sim) or a random-FOV pattern. Visible hits are scattered into the
  grid with :func:`ame2.mapping.grid.points_to_local_grid` (max z, holes = -3),
  so occlusion is geometric. The remaining augmentations (uniform noise with
  random magnitude, border crops, clipping ranges, missing points, outliers) are
  applied on the grid by :func:`ame2.mapping.augment.augment_inputs`.

The grid convention matches :class:`ame2.sensors.LivoxNeuralMapSensor`: centred
on the sensor xy, axes = base yaw, heights relative to the sensor z.

Everything stays on the torch device (``wp.from_torch``), nothing is written to disk.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .augment import AugmentCfg, augment_inputs
from .grid import LocalGridSpec, points_to_local_grid
from .terrain_meshes import FAMILIES, TerrainMesh, build_family, load_terrain_npz

try:
    import warp as wp
except ImportError as e:  # pragma: no cover
    raise ImportError("ame2.mapping.warp_sampler needs `pip install warp-lang`") from e

MID360_RAYDIRS = os.path.join(os.path.dirname(__file__), "..", "sensors", "mid360_raydirs.npy")
# Mid360 hangs upside down with a calibrated pitch (see sensors/mid360_lidar.py):
# sensor -> base rotation ~ Ry(0.0511) @ Rx(pi) (torso == base at the default pose).
_MID360_PITCH = 0.05112069379091391

_wp_ready = False


def _init_warp():
    global _wp_ready
    if not _wp_ready:
        wp.config.quiet = True
        wp.init()
        _wp_ready = True


@wp.kernel
def _raycast_kernel(
    mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),     # (B,)
    dirs: wp.array(dtype=wp.vec3),        # (B*R,)
    rays_per_frame: int,
    max_dist: float,
    hits: wp.array(dtype=wp.vec3),        # (B*R,)
):
    i = wp.tid()
    o = origins[i / rays_per_frame]
    d = dirs[i]
    q = wp.mesh_query_ray(mesh, o, d, max_dist)
    if q.result:
        hits[i] = o + d * q.t
    else:
        nan = wp.nan
        hits[i] = wp.vec3(nan, nan, nan)


@wp.kernel
def _topdown_kernel(
    mesh: wp.uint64,
    xy: wp.array(dtype=wp.vec2),          # (N,) world xy of each ray
    z_start: wp.array(dtype=float),       # (N,) start height of each ray
    max_dist: float,
    z_out: wp.array(dtype=float),         # (N,) hit z or nan
):
    i = wp.tid()
    p = xy[i]
    z0 = z_start[i]
    q = wp.mesh_query_ray(mesh, wp.vec3(p[0], p[1], z0), wp.vec3(0.0, 0.0, -1.0), max_dist)
    if q.result:
        z_out[i] = z0 - q.t
    else:
        z_out[i] = wp.nan


def raycast(mesh: "wp.Mesh", origins: torch.Tensor, dirs: torch.Tensor, max_dist: float = 30.0) -> torch.Tensor:
    """origins (B,3), dirs (B,R,3) unit -> hits (B,R,3), nan for misses."""
    B, R, _ = dirs.shape
    dev = str(origins.device)
    o = wp.from_torch(origins.contiguous().float(), dtype=wp.vec3)
    d = wp.from_torch(dirs.reshape(-1, 3).contiguous().float(), dtype=wp.vec3)
    out = torch.empty(B * R, 3, device=origins.device)
    wp.launch(_raycast_kernel, dim=B * R, inputs=[mesh.id, o, d, R, float(max_dist)],
              outputs=[wp.from_torch(out, dtype=wp.vec3)], device=dev)
    return out.view(B, R, 3)


def topdown(mesh: "wp.Mesh", xy: torch.Tensor, z_start, max_dist: float = 100.0) -> torch.Tensor:
    """xy (..., 2) -> first-hit z of a downward ray from ``z_start`` (float or tensor
    broadcastable to xy[..., 0]); nan where nothing is hit."""
    shape = xy.shape[:-1]
    flat = xy.reshape(-1, 2).contiguous().float()
    zs = torch.as_tensor(z_start, dtype=torch.float32, device=xy.device).expand(shape).reshape(-1).contiguous()
    out = torch.empty(flat.shape[0], device=xy.device)
    wp.launch(_topdown_kernel, dim=flat.shape[0], inputs=[mesh.id, wp.from_torch(flat, dtype=wp.vec2),
              wp.from_torch(zs), float(max_dist)], outputs=[wp.from_torch(out)], device=str(xy.device))
    return out.view(shape)


def _rot_zyx(yaw, pitch, roll):
    """(B,) angles -> (B,3,3) R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy, cp, sp, cr, sr = yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin(), roll.cos(), roll.sin()
    R = torch.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    ], -1)
    return R.view(-1, 3, 3)


@dataclass
class SamplerCfg:
    families: tuple[str, ...] = FAMILIES
    family_weights: tuple[float, ...] = (0.4, 0.2, 0.2, 0.2)   # locomotion, stacked, heightfield, floating
    locomotion_npz: str | None = None          # from export_terrain_mesh.py; family skipped if missing
    procedural_size: float = 40.0              # m, side of each procedural mesh
    remesh_every: int = 0                      # rebuild procedural meshes every N batches (0 = never)
    # robot pose (G1): Mid360 height above the ground below it, small base tilt
    sensor_height: tuple[float, float] = (0.9, 1.45)
    max_tilt_deg: float = 10.0
    # input ray pattern
    rays_per_frame: int = 20000               # = one Mid360 window in sim (LivoxNeuralMapSensor)
    mid360_prob: float = 0.5                  # else random-FOV pattern
    hfov_deg: tuple[float, float] = (90.0, 360.0)
    vfov_low_deg: tuple[float, float] = (-90.0, -25.0)
    vfov_high_deg: tuple[float, float] = (-5.0, 55.0)
    sensor_xy_jitter: float = 0.1             # sensor pose noise vs. the grid origin
    sensor_z_jitter: float = 0.05
    max_range: float = 20.0
    # GT rays start this far above the sensor (Isaac's mapping_gt_scanner: base + 3 m ~ sensor + 2.5 m),
    # so overhangs higher than that are "seen through", as in sim
    gt_start_above_sensor: float = 2.5
    clean_prob: float = 0.0                   # fraction of frames without grid augmentations
    aug: AugmentCfg = field(default_factory=AugmentCfg)


class WarpMappingSampler:
    """``sample(B) -> (x, y, mask)`` each (B,1,L,W): corrupted input, GT label, label validity."""

    def __init__(self, spec: LocalGridSpec, cfg: SamplerCfg = SamplerCfg(), device: str = "cuda", seed: int = 0,
                 meshes: list[TerrainMesh] | None = None):
        _init_warp()
        self.spec, self.cfg = spec, cfg
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self._calls = 0
        X, Y = spec.axes(self.device)
        self._cells = torch.stack([X, Y], -1)               # (L, W, 2) grid-frame cell centres
        dirs = np.load(MID360_RAYDIRS).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        self._mid360 = torch.from_numpy(dirs).to(self.device)
        mount = _rot_zyx(torch.zeros(1), torch.tensor([_MID360_PITCH]), torch.tensor([math.pi]))[0]
        self._mid360_mount = mount.to(self.device)
        if meshes is not None:
            self.meshes = list(meshes)
            self.weights = np.ones(len(self.meshes)) / len(self.meshes)
        else:
            self._build_meshes()
        self._upload()

    # -- meshes ---------------------------------------------------------------------------
    def _build_meshes(self):
        self.meshes, w = [], []
        for name, wt in zip(self.cfg.families, self.cfg.family_weights):
            if wt <= 0:
                continue
            if name == "locomotion":
                p = self.cfg.locomotion_npz
                if not p or not os.path.isfile(p):
                    print(f"[WarpMappingSampler] no locomotion mesh ({p!r}); export one with "
                          "ame2/scripts/mapping/export_terrain_mesh.py -- using procedural families only")
                    continue
                self.meshes.append(load_terrain_npz(p))
            else:
                self.meshes.append(build_family(name, self.rng, size=self.cfg.procedural_size))
            w.append(wt)
        if not self.meshes:
            raise RuntimeError("no terrain meshes")
        self.weights = np.asarray(w, np.float64) / np.sum(w)

    def _upload(self):
        dev = str(self.device)
        self._wp_meshes = []
        for m in self.meshes:
            pts = wp.array(m.vertices, dtype=wp.vec3, device=dev)
            idx = wp.array(m.faces.reshape(-1), dtype=wp.int32, device=dev)
            self._wp_meshes.append(wp.Mesh(points=pts, indices=idx))
        self._ztop = [float(m.vertices[:, 2].max()) + 1.0 for m in self.meshes]

    def regenerate(self):
        keep = [m for m in self.meshes if m.name == "locomotion"]
        self._build_meshes()
        self.meshes = keep + [m for m in self.meshes if m.name != "locomotion"]
        self._upload()

    # -- sampling -------------------------------------------------------------------------
    def _rand(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=self.device, generator=self.gen)

    def _poses(self, k: int, mi: int):
        m = self.meshes[mi]
        margin = max(abs(self.spec.x_range[0]), abs(self.spec.x_range[1]), abs(self.spec.y_range[1])) + 0.5
        if m.origins is not None and len(m.origins) > 0:
            o = torch.from_numpy(m.origins).to(self.device)
            j = torch.randint(0, o.shape[0], (k,), device=self.device, generator=self.gen)
            # anywhere within ~ a sub-terrain around its origin
            xy = o[j, :2] + torch.stack([self._rand(k, -5.0, 5.0), self._rand(k, -5.0, 5.0)], -1)
        else:
            x0, x1, y0, y1 = m.bounds
            xy = torch.stack([self._rand(k, x0 + margin, x1 - margin), self._rand(k, y0 + margin, y1 - margin)], -1)
        ground = topdown(self._wp_meshes[mi], xy, m.probe_z)
        ground = torch.nan_to_num(ground, nan=0.0)
        z = ground + self._rand(k, *self.cfg.sensor_height)
        yaw = self._rand(k, -math.pi, math.pi)
        return torch.cat([xy, z[:, None]], -1), yaw

    def _ray_dirs(self, k: int, yaw: torch.Tensor):
        c, R = self.cfg, self.cfg.rays_per_frame
        t = math.radians(c.max_tilt_deg)
        rot = _rot_zyx(yaw, self._rand(k, -t, t), self._rand(k, -t, t))       # base orientation
        # Mid360: contiguous random window of the real scan sequence, real mount
        n = self._mid360.shape[0]
        start = torch.randint(0, n, (k, 1), device=self.device, generator=self.gen)
        idx = (start + torch.arange(R, device=self.device)[None]) % n
        d_mid = self._mid360[idx] @ self._mid360_mount.T                       # (k,R,3) base frame
        # random FOV: azimuth sector + elevation band, random centre direction
        hf = torch.deg2rad(self._rand(k, *c.hfov_deg))[:, None]
        az = self._rand(k, -0.35 * math.pi, 0.35 * math.pi)[:, None]   # sector centre, mostly forward
        lo = torch.deg2rad(self._rand(k, *c.vfov_low_deg))[:, None]
        hi = torch.deg2rad(self._rand(k, *c.vfov_high_deg))[:, None]
        u = torch.rand(k, R, device=self.device, generator=self.gen)
        v = torch.rand(k, R, device=self.device, generator=self.gen)
        a = az + (u - 0.5) * hf
        s_lo, s_hi = torch.sin(lo), torch.sin(hi)
        sz = s_lo + (s_hi - s_lo) * v                                           # uniform on the sphere band
        cz = torch.sqrt((1 - sz * sz).clamp_min(0))
        d_fov = torch.stack([cz * a.cos(), cz * a.sin(), sz], -1)
        use_mid = (torch.rand(k, device=self.device, generator=self.gen) < c.mid360_prob).view(k, 1, 1)
        d = torch.where(use_mid, d_mid, d_fov)
        return torch.einsum("kij,krj->kri", rot, d)

    @torch.no_grad()
    def render(self, mi: int, pos: torch.Tensor, yaw: torch.Tensor, clean: bool = False):
        """Frames at given sensor poses on mesh ``mi``: pos (k,3) Mid360 position, yaw (k,).

        Returns (x, y, label_ok), each (k, L, W).
        """
        spec, c = self.spec, self.cfg
        k = pos.shape[0]
        pos, yaw = pos.to(self.device).float(), yaw.to(self.device).float()
        # ground truth: top-down ray per cell (grid frame: sensor xy, base yaw)
        cy, sy = yaw.cos().view(k, 1, 1), yaw.sin().view(k, 1, 1)
        gx = pos[:, 0].view(k, 1, 1) + cy * self._cells[..., 0] - sy * self._cells[..., 1]
        gy = pos[:, 1].view(k, 1, 1) + sy * self._cells[..., 0] + cy * self._cells[..., 1]
        z0 = (pos[:, 2] + c.gt_start_above_sensor).view(k, 1, 1)
        zt = topdown(self._wp_meshes[mi], torch.stack([gx, gy], -1), z0)
        label_ok = torch.isfinite(zt)
        y = torch.where(label_ok, (zt - pos[:, 2].view(k, 1, 1)).clamp(spec.hole_value, spec.z_max),
                        torch.full_like(zt, spec.hole_value))
        # input: rays from a perturbed sensor pose -> visible hits -> max-z grid
        jit = torch.stack([self._rand(k, -c.sensor_xy_jitter, c.sensor_xy_jitter),
                           self._rand(k, -c.sensor_xy_jitter, c.sensor_xy_jitter),
                           self._rand(k, -c.sensor_z_jitter, c.sensor_z_jitter)], -1)
        hits = raycast(self._wp_meshes[mi], pos + jit, self._ray_dirs(k, yaw), c.max_range)
        x = points_to_local_grid(hits, pos, yaw, spec)
        if not clean:
            aug = augment_inputs(x, spec, c.aug, generator=self.gen)
            if c.clean_prob > 0:
                keep = (torch.rand(k, device=self.device, generator=self.gen) < c.clean_prob).view(k, 1, 1)
                aug = torch.where(keep, x, aug)
            x = aug
        return x, y, label_ok

    def family_index(self, name: str) -> int:
        for i, m in enumerate(self.meshes):
            if m.name == name:
                return i
        raise KeyError(f"mesh family {name!r} not loaded ({[m.name for m in self.meshes]})")

    @torch.no_grad()
    def sample(self, batch: int, clean: bool = False, family: str | None = None):
        """Random frames; ``family`` restricts sampling to one mesh family."""
        self._calls += 1
        if self.cfg.remesh_every and self._calls % self.cfg.remesh_every == 0:
            self.regenerate()
        if family is not None:
            counts = np.zeros(len(self.meshes), dtype=np.int64)
            counts[self.family_index(family)] = batch
        else:
            counts = np.bincount(self.rng.choice(len(self.meshes), size=batch, p=self.weights),
                                 minlength=len(self.meshes))
        xs, ys, ms = [], [], []
        for mi, k in enumerate(counts):
            if k == 0:
                continue
            pos, yaw = self._poses(int(k), mi)
            x, y, ok = self.render(mi, pos, yaw, clean=clean)
            xs.append(x), ys.append(y), ms.append(ok)
        perm = torch.randperm(batch, device=self.device, generator=self.gen)
        if family is not None:
            perm = torch.arange(batch, device=self.device)
        return (torch.cat(xs)[perm].unsqueeze(1), torch.cat(ys)[perm].unsqueeze(1),
                torch.cat(ms)[perm].unsqueeze(1))
