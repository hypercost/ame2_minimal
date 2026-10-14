"""Warp ray-tracing data sampler (CPU warp): GT heights, geometric occlusion, augment ranges, training."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

wp = pytest.importorskip("warp")

from ame2.mapping import terrain_meshes as tm  # noqa: E402
from ame2.mapping.augment import AugmentCfg, augment_inputs  # noqa: E402
from ame2.mapping.grid import LocalGridSpec, points_to_local_grid  # noqa: E402
from ame2.mapping.warp_sampler import SamplerCfg, WarpMappingSampler, raycast, topdown  # noqa: E402

SPEC = LocalGridSpec((-3.0, 3.0), (-1.5, 1.5), 0.03)      # the G1 sensor grid
SMALL = LocalGridSpec((-1.5, 1.5), (-0.75, 0.75), 0.05)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _box_on_plane():
    plane = tm.plane_mesh(-20, 20, -20, 20, 0.0)
    # 0.5 m high box, x in [1, 1.6], y in [-1, 1]
    box = tm.boxes_mesh(np.array([[1.3, 0.0]]), np.array([[0.6, 2.0]]), np.array([0.0]), np.array([0.5]),
                        np.array([0.0]))
    v, f = tm.merge(plane, box)
    return tm.TerrainMesh("box", v, f, (-5, 5, -5, 5), probe_z=5.0)


def _wp_mesh(m):
    return wp.Mesh(points=wp.array(m.vertices, dtype=wp.vec3, device="cpu"),
                   indices=wp.array(m.faces.reshape(-1), dtype=wp.int32, device="cpu"))


def test_topdown_gt_heights_box_on_plane():
    WarpMappingSampler  # init warp via the module import
    from ame2.mapping.warp_sampler import _init_warp

    _init_warp()
    mesh = _wp_mesh(_box_on_plane())
    xy = torch.tensor([[0.0, 0.0], [1.3, 0.0], [1.3, 0.9], [1.3, 1.1], [2.0, 0.0], [50.0, 0.0]])
    z = topdown(mesh, xy, 3.0)
    assert torch.allclose(z[:5], torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0]), atol=1e-5)
    assert torch.isnan(z[5])  # off the mesh


def test_geometric_occlusion_behind_box():
    from ame2.mapping.warp_sampler import _init_warp

    _init_warp()
    mesh = _wp_mesh(_box_on_plane())
    sensor = torch.tensor([[0.0, 0.0, 1.0]])
    # dense rays over the lower hemisphere
    n = 200_000
    g = torch.Generator().manual_seed(0)
    a = 2 * math.pi * torch.rand(n, generator=g)
    sz = -torch.rand(n, generator=g)
    cz = torch.sqrt(1 - sz**2)
    dirs = torch.stack([cz * a.cos(), cz * a.sin(), sz], -1)[None]
    hits = raycast(mesh, sensor, dirs, 20.0)
    grid = points_to_local_grid(hits, sensor, torch.zeros(1), SPEC)[0]
    X, Y = SPEC.axes()
    band = Y.abs() < 0.5
    # in front of the box: ground at -1 m (sensor-relative)
    def seen(region, value):
        g = grid[region]
        obs = g > SPEC.hole_value
        assert obs.float().mean() > 0.8              # random rays leave a few empty cells
        assert (g[obs] - value).abs().max() < 1e-3

    seen(band & (X > 0.3) & (X < 0.9), -1.0)
    # box top visible: -0.5
    seen(band & (X > 1.05) & (X < 1.55), -0.5)
    # right behind the box: shadow (ray to (x, 0) passes the box edge at x=1.6, z=-0.5 for x < 3.2)
    shadow = band & (X > 1.75) & (X < 2.9)
    assert (grid[shadow] == SPEC.hole_value).float().mean() > 0.99
    # beside the box (|y| > 1.2 at small x): visible ground
    side = (Y.abs() > 1.25) & (X > -1.0) & (X < 0.5)
    assert (grid[side] > SPEC.hole_value).float().mean() > 0.95


def test_sampler_box_scene_labels_and_shapes():
    s = WarpMappingSampler(SPEC, SamplerCfg(sensor_height=(1.0, 1.0), max_tilt_deg=0.0), device="cpu",
                           meshes=[_box_on_plane()])
    x, y, m = s.sample(8, clean=True)
    assert x.shape == y.shape == m.shape == (8, 1, *SPEC.shape)
    assert m.all()
    # labels are sensor-relative ground (-1) or box top (-0.5) (or plane at -1 under a raised probe)
    vals = torch.unique(torch.round(y * 100) / 100)
    assert set(vals.tolist()) <= {-1.0, -0.5, -1.5}
    # clean input agrees with the label wherever something was observed (geometry is shared)
    obs = x > SPEC.hole_value + 0.1
    assert obs.float().mean() > 0.05
    err = (x - y).abs()[obs]
    assert (err < 0.02).float().mean() > 0.97   # mismatches only at box edges (side-wall hits)


def test_procedural_families_and_augmented_sample():
    cfg = SamplerCfg(procedural_size=12.0, rays_per_frame=4000)
    s = WarpMappingSampler(SMALL, cfg, device="cpu", seed=3)
    assert {m.name for m in s.meshes} == {"stacked_boxes", "heightfield", "floating_boxes"}
    for mesh in s.meshes:
        assert mesh.vertices.dtype == np.float32 and mesh.faces.max() < mesh.vertices.shape[0]
    x, y, m = s.sample(16)
    assert x.shape == (16, 1, *SMALL.shape)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert x.min() >= SMALL.hole_value and x.max() <= SMALL.z_max
    assert y.min() >= SMALL.hole_value and y.max() <= SMALL.z_max
    assert m.float().mean() > 0.95
    s.regenerate()
    assert len(s.meshes) == 3


def test_locomotion_npz_roundtrip(tmp_path):
    v, f = tm.heightfield_mesh(np.zeros((50, 50), np.float32), 0.2, -5.0, -5.0)
    p = tmp_path / "terrain.npz"
    np.savez(p, vertices=v, faces=f, origins=np.zeros((1, 1, 3), np.float32), bounds=np.array([-5, 5, -5, 5.0]))
    mesh = tm.load_terrain_npz(str(p))
    assert mesh.name == "locomotion" and mesh.origins.shape == (1, 3)
    cfg = SamplerCfg(locomotion_npz=str(p), family_weights=(1.0, 0.0, 0.0, 0.0), rays_per_frame=2000)
    s = WarpMappingSampler(SMALL, cfg, device="cpu")
    x, y, m = s.sample(4, clean=True)
    assert [mm.name for mm in s.meshes] == ["locomotion"]
    assert ((y[m] + 0.0) <= -0.9 + 1e-4).all()   # flat ground >= 0.9 m below the sensor


def test_augment_ranges():
    g = torch.Generator().manual_seed(0)
    x = torch.full((64, *SMALL.shape), -1.0)
    x[:, :, :5] = SMALL.hole_value
    cfg = AugmentCfg()
    out = augment_inputs(x, SMALL, cfg, generator=g)
    assert out.shape == x.shape
    assert out.min() >= SMALL.hole_value and out.max() <= SMALL.z_max
    obs = out > SMALL.hole_value
    # noise stays within the per-sample magnitude except outliers / clipping
    near = (out[obs] + 1.0).abs() <= cfg.noise_max + 1e-6
    assert near.float().mean() > 0.9
    assert (~obs[:, :, 5:]).float().mean() > 0.0          # missing points / crops happened
    lo, hi = cfg.outlier_range
    far = out[obs][~near]
    assert ((far >= lo - 1e-6) & (far <= hi + 1e-6)).all()


def test_train_on_warp_generator_cpu(tmp_path):
    path = os.path.join(REPO, "ame2", "scripts", "mapping", "train_mapping_model.py")
    spec = importlib.util.spec_from_file_location("train_mapping_model_w", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hist = mod.main([
        "--source", "warp", "--frames", "160", "--batch_size", "16", "--clean_frames", "32",
        "--rays_per_frame", "3000", "--x_range", "-1.5", "1.5", "--y_range", "-0.75", "0.75", "--res", "0.05",
        "--device", "cpu", "--out_dir", str(tmp_path / "out"), "--export", "", "--log_every", "5", "--val_every", "10",
        "--remesh_every", "0",
    ])
    assert len(hist) == 10 and all(math.isfinite(h) for h in hist)
    assert (tmp_path / "out" / "mapping_model.jit").is_file()


def test_mapping_imports_without_kit(tmp_path):
    """An importable-but-not-started Isaac Lab must not break the offline tools."""
    fake = tmp_path / "fake"
    (fake / "isaaclab").mkdir(parents=True)
    (fake / "isaaclab" / "__init__.py").write_text("raise ImportError('omni not available: Kit not started')\n")
    code = (
        "import importlib.util, sys\n"
        "assert importlib.util.find_spec('isaaclab') is not None\n"
        "import ame2, ame2.sensors, ame2.mapping, ame2.mapping.warp_sampler\n"
        "from ame2.sensors.neural_mapping_utils import LivoxGlobalProbWinnerPipeline\n"
        "spec = importlib.util.spec_from_file_location('t', sys.argv[1]); m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m); print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(fake), os.path.join(REPO, "ame2")]))
    r = subprocess.run([sys.executable, "-c", code, os.path.join(REPO, "ame2", "scripts", "mapping",
                        "train_mapping_model.py")], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_cpu_throughput_report(capsys):
    """Not an assertion on speed: prints the measured CPU sampler throughput (G1 grid)."""
    s = WarpMappingSampler(SPEC, SamplerCfg(procedural_size=20.0), device="cpu", seed=0)
    s.sample(8)
    t = time.time()
    s.sample(64)
    fps = 64 / (time.time() - t)
    with capsys.disabled():
        print(f"\n[CPU warp sampler, {SPEC.shape} grid, 20k rays/frame] {fps:.1f} frames/s on {os.cpu_count()} cores")
    assert fps > 0
