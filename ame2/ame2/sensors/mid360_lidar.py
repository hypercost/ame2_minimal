"""Livox Mid360 lidar simulation for the G1.

Reproduces the ray pattern used to train the lidar height-map model: the
bundled table holds the real Mid360 non-repetitive scan sequence (~800k unit
directions, sensor frame); each scan uses one random contiguous window of
``rays_per_scan`` rows (one 0.1 s scan at 10 Hz), shared across all envs
updated in the same call.

Mid360 sensor artifacts are only modelled crudely: with
``Mid360RayCasterCfg.enable_noise`` :func:`add_noise` drops a fraction of the
points and pulls another fraction towards the sensor along the ray (paper
Appendix B: 15% missing points, 2% artifacts for student training). Real
Mid360 effects (mixed-pixel ghosts, edge dropout, driver filter, near-field
pull-ins, ...) are not reproduced.
"""

from __future__ import annotations

import math
import os
import numpy as np
import torch
from collections.abc import Sequence
from typing import Callable, Tuple

import warp as wp

from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from isaaclab.utils.warp import ProxyArray, raycast_mesh

from ame2.mapping.pointcloud_noise import apply_pointcloud_noise

# Bundled Mid360 scan sequence: (N, 3) fp32 unit ray directions in the sensor
# frame, in real scan order.
_MID360_RAYDIR_PATH = os.path.join(os.path.dirname(__file__), "mid360_raydirs.npy")

# Mounting orientation: R_tot = R_base @ R_bias with R_bias = Ry(pitch) @ Rx(roll)
# (R = Rz@Ry@Rx); the Mid360 hangs upside down (roll pi) with a calibrated pitch.
_SENSOR_BIAS_ROLL = math.pi
_SENSOR_BIAS_PITCH = 0.05112069379091391
# In g1_lidar.usd the mid360_joint already pitches mid360_link w.r.t. the
# torso by Ry(2*atan2(0.02006994, 0.9997986)) (localRot0 of the fixed joint),
# so the sensor-frame offset applied on mid360_link only carries the flip and
# the residual calibrated-vs-nominal pitch. Assumes torso_link == base at the
# default waist pose (as during calibration).
_USD_MID360_JOINT_PITCH = 2.0 * math.atan2(0.02006994, 0.9997986)


def _mid360_mount_offset_rot() -> Tuple[float, float, float, float]:
    """Quaternion (x,y,z,w) of Ry(bias_pitch - usd_joint_pitch) @ Rx(pi).

    IsaacLab 3.0 uses XYZW quaternions everywhere (2.x: WXYZ); the product is computed in WXYZ
    below and returned reordered.
    """
    half_p = 0.5 * (_SENSOR_BIAS_PITCH - _USD_MID360_JOINT_PITCH)
    qy = (math.cos(half_p), 0.0, math.sin(half_p), 0.0)
    qx = (math.cos(0.5 * _SENSOR_BIAS_ROLL), math.sin(0.5 * _SENSOR_BIAS_ROLL), 0.0, 0.0)
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qx
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return (x, y, z, w)


MID360_MOUNT_OFFSET_ROT: Tuple[float, float, float, float] = _mid360_mount_offset_rot()
"""``RayCasterCfg.OffsetCfg.rot`` for a Mid360 sensor parented to ``mid360_link``."""


def add_noise(
    ray_hits_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    enable: bool = False,
    drop_ratio: float = 0.15,
    artifact_ratio: float = 0.02,
) -> torch.Tensor:
    """Mid360 point corruption (see module docstring).

    Args:
        ray_hits_w: (E, R, 3) world-frame hit points (misses = inf).
        sensor_pos_w: (E, 3) world sensor origins.
        ray_dirs_w: (E, R, 3) world ray directions (unused by the current model).
        enable: master switch; False (default) returns the hits unchanged.
        drop_ratio: fraction of points turned into misses.
        artifact_ratio: fraction of points moved towards the sensor along the ray.
    """
    if not enable:
        return ray_hits_w
    return apply_pointcloud_noise(ray_hits_w, sensor_pos_w, drop_ratio=drop_ratio, artifact_ratio=artifact_ratio)


def mid360_pattern(cfg: "Mid360PatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static fallback pattern: the first window of the scan sequence.

    :class:`Mid360RayCaster` re-slices a random window per update; this function
    only fixes the ray count (and lets a plain ``RayCaster`` use a frozen window).
    """
    dirs = cfg.load_table(device)
    window = dirs[: cfg.rays_per_scan : cfg.ray_decimation]
    return torch.zeros_like(window), window.clone()


@configclass
class Mid360PatternCfg(PatternBaseCfg):
    """Rolling-window pattern over the real Mid360 scan sequence."""

    func: Callable = mid360_pattern

    raydirs_path: str = _MID360_RAYDIR_PATH
    """Path to the baked (N, 3) fp32 direction table (sensor frame, scan order)."""

    rays_per_scan: int = 20000
    """Contiguous rows consumed per scan (real Mid360: 200k pts/s @ 10 Hz = 20k)."""

    ray_decimation: int = 1
    """Keep every n-th ray inside the window (same spatial coverage, thinner
    density). 1 = the real ray count."""

    def load_table(self, device: str) -> torch.Tensor:
        """Load (cached) and return the full normalized direction table."""
        if not hasattr(self, "_all_dirs"):
            dirs = torch.from_numpy(np.load(self.raydirs_path).astype(np.float32))
            self._all_dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            assert self._all_dirs.shape[0] >= self.rays_per_scan, (
                f"Mid360 table has {self._all_dirs.shape[0]} rows < rays_per_scan={self.rays_per_scan}"
            )
        if str(self._all_dirs.device) != str(device):
            self._all_dirs = self._all_dirs.to(device)
        return self._all_dirs


class _Mid360Mixin:
    """Mid360 behaviour layered on top of the backend ray-caster (IsaacLab 3.0).

    In 3.0 ``isaaclab.sensors.RayCaster`` is a factory that instantiates the backend class
    (PhysX / OVPhysX / Newton) at construction time, so :class:`Mid360RayCaster` builds a
    subclass ``(_Mid360Mixin, <backend RayCaster>)`` on the fly. Backend pose tracking
    (``_get_view_transforms_wp`` + ``_offset_pos_wp`` / ``_offset_quat_wp``) is reused as is.

    Deviations from the base class (for the 20k-rays x many-envs scale):

    * ray directions are stored once in the full table (no per-env ``repeat``);
    * rays always follow the full body orientation (``ray_alignment`` is ignored);
    * per-ray world buffers (``_ray_starts_w`` / ``_ray_directions_w``) are only kept when
      ``cfg.keep_world_ray_buffers`` is set (play.py --viz_lidar).
    """

    cfg: "Mid360RayCasterCfg"

    def _initialize_rays_impl(self):
        pattern: Mid360PatternCfg = self.cfg.pattern_cfg
        n_env = self._view_count
        # full table with the mounting rotation baked in (world dirs then only need the body quat)
        all_dirs = pattern.load_table(self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self._all_ray_dirs = quat_apply(offset_quat.expand(all_dirs.shape[0], 4), all_dirs)
        self._offset_pos_cfg = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        self._n_total_dirs = self._all_ray_dirs.shape[0]
        self._window_len = pattern.rays_per_scan
        self._decimation = pattern.ray_decimation
        self.num_rays = len(range(0, self._window_len, self._decimation))
        # base-class buffers used by reset()
        self.drift = ProxyArray(wp.zeros(n_env, dtype=wp.vec3f, device=self._device))
        self.ray_cast_drift = ProxyArray(wp.zeros(n_env, dtype=wp.vec3f, device=self._device))
        self._data.create_buffers(n_env, self.num_rays, self._device)
        self._ray_starts_w_t = None
        self._ray_directions_w_t = None
        if self.cfg.keep_world_ray_buffers:
            self._ray_starts_w_t = torch.zeros(n_env, self.num_rays, 3, device=self._device)
            self._ray_directions_w_t = torch.zeros(n_env, self.num_rays, 3, device=self._device)

    def _sensor_pose_torch(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose (pos, quat xyzw) of the sensor frame (before ``cfg.offset.rot``)."""
        tf = wp.to_torch(self._get_view_transforms_wp()).reshape(-1, 7)[env_ids]
        off_pos = wp.to_torch(self._offset_pos_wp).reshape(-1, 3)[env_ids]
        off_quat = wp.to_torch(self._offset_quat_wp).reshape(-1, 4)[env_ids]
        pos_w, quat_w = combine_frame_transforms(tf[:, :3], tf[:, 3:7], off_pos, off_quat)
        pos_w = pos_w + quat_apply(quat_w, self._offset_pos_cfg.expand(pos_w.shape[0], 3))
        return pos_w, quat_w

    def _update_buffers_impl(self, env_mask: wp.array):
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        # one random contiguous window per call, shared by the updated envs
        start = int(torch.randint(0, self._n_total_dirs - self._window_len + 1, (1,)).item())
        dirs = self._all_ray_dirs[start : start + self._window_len : self._decimation]  # (R, 3)

        pos_w, quat_w = self._sensor_pose_torch(env_ids)
        pos_w = pos_w + self.drift.torch[env_ids]
        wp.to_torch(self._data._pos_w)[env_ids] = pos_w
        wp.to_torch(self._data._quat_w)[env_ids] = quat_w

        n = pos_w.shape[0]
        ray_dirs_w = quat_apply(quat_w.repeat(1, self.num_rays), dirs.unsqueeze(0).expand(n, self.num_rays, 3))
        ray_starts_w = pos_w.unsqueeze(1).expand(n, self.num_rays, 3).contiguous()

        mesh = BaseRayCaster.meshes[(self.cfg.mesh_prim_paths[0], self._device)]
        hits = raycast_mesh(ray_starts_w, ray_dirs_w.contiguous(), max_dist=self.cfg.max_distance, mesh=mesh)[0]
        hits[:, :, 2] += self.ray_cast_drift.torch[env_ids, 2].unsqueeze(-1)
        hits = add_noise(
            hits, pos_w, ray_dirs_w, enable=self.cfg.enable_noise,
            drop_ratio=self.cfg.noise_drop_ratio, artifact_ratio=self.cfg.noise_artifact_ratio,
        )
        wp.to_torch(self._data._ray_hits_w)[env_ids] = hits

        if self._ray_starts_w_t is not None:
            self._ray_starts_w_t[env_ids] = ray_starts_w
            self._ray_directions_w_t[env_ids] = ray_dirs_w


_MID360_CLASS_CACHE: dict[type, type] = {}


class Mid360RayCaster:
    """Ray-caster casting a fresh random window of the Mid360 sequence per update.

    Factory: returns an instance of ``(_Mid360Mixin, <active backend RayCaster>)``.
    """

    def __new__(cls, cfg: "Mid360RayCasterCfg"):
        base = RayCaster.resolve_class(cfg)
        impl = _MID360_CLASS_CACHE.get(base)
        if impl is None:
            impl = type(f"Mid360RayCaster_{base.__module__.split('.')[0]}", (_Mid360Mixin, base), {})
            _MID360_CLASS_CACHE[base] = impl
        return impl(cfg)


@configclass
class Mid360RayCasterCfg(RayCasterCfg):
    """Configuration for :class:`Mid360RayCaster`."""

    class_type: type = Mid360RayCaster

    pattern_cfg: Mid360PatternCfg = Mid360PatternCfg()

    offset: RayCasterCfg.OffsetCfg = RayCasterCfg.OffsetCfg(rot=MID360_MOUNT_OFFSET_ROT)
    """Sensor frame in the parent (``mid360_link``) frame: the upside-down
    mount flip plus the residual calibrated pitch (see module docstring)."""

    max_distance: float = 6.0
    """Max lidar range (m). The EM input grid only spans +-3 m around the
    sensor (slant ranges < ~5 m), so 6 m covers everything the map can use."""

    enable_noise: bool = False
    """Master switch for :func:`add_noise` (point dropout + artifacts)."""

    noise_drop_ratio: float = 0.15
    """Fraction of points dropped when ``enable_noise`` (paper App. B: 15%)."""

    noise_artifact_ratio: float = 0.02
    """Fraction of artifact points when ``enable_noise`` (paper App. B: 2%)."""

    keep_world_ray_buffers: bool = False
    """Allocate per-ray world start/direction buffers (2 x E x R x 3 floats —
    ~3.6 GB at 8k envs x 20k rays). Needed only by play.py --viz_lidar."""
