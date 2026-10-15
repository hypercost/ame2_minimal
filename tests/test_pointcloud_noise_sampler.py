"""Regression: the Warp sampler must reproduce the sim lidar's along-ray artifacts (diag_simgap: +0.13 m
in-loop bias from artifacts the model never saw), and clean frames must stay artifact-free."""
import math

import pytest
pytest.importorskip("warp")
import numpy as np
import torch

from ame2.mapping.grid import LocalGridSpec
from ame2.mapping.terrain_meshes import TerrainMesh
from ame2.mapping.warp_sampler import SamplerCfg, WarpMappingSampler


def _plane(size=20.0):
    v = np.array([[-size, -size, 0], [size, -size, 0], [size, size, 0], [-size, size, 0]], np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    return TerrainMesh("plane", v, f, (-size, size, -size, size), probe_z=5.0)


def _render(cfg, clean):
    s = WarpMappingSampler(LocalGridSpec(), cfg, device="cpu", seed=0, meshes=[_plane()])
    pos = torch.tensor([[0.0, 0.0, 1.2]]).repeat(16, 1)
    yaw = torch.zeros(16)
    return s.render(0, pos, yaw, clean=clean)


def test_clean_frames_have_no_artifacts():
    x, y, ok = _render(SamplerCfg(mid360_prob=1.0, max_tilt_deg=0.0, sensor_z_jitter=0.0), clean=True)
    m = (x > -2.9) & ok
    assert (x - y)[m].abs().max() < 1e-3


def test_artifacts_land_above_ground():
    cfg = SamplerCfg(mid360_prob=1.0, max_tilt_deg=0.0, sensor_z_jitter=0.0, pc_noise_prob=1.0,
                     pc_artifact_max=0.04, pc_drop_max=0.0)
    cfg.aug.noise_max = 0.0
    cfg.aug.outlier_max = 0.0
    cfg.aug.clip_prob = 0.0
    torch.manual_seed(0)
    x, y, ok = _render(cfg, clean=False)
    m = (x > -2.9) & ok
    assert ((x - y)[m] > 0.2).sum() > 50   # along-ray artifacts float above the plane
