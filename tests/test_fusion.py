import math

import torch

from ame2.mapping.grid import LocalGridSpec, points_to_local_grid
from ame2.mapping.models import DummyMappingModel
from ame2.sensors.neural_mapping_utils import LIVOX_HOLE, GlobalGrid, LivoxGlobalProbWinnerPipeline


def _reference_recenter(grid: GlobalGrid, mask, cx, cy, fill_world=None):
    """The original per-env python-loop implementation (torch.roll + stripe fill)."""
    res = torch.as_tensor(grid.res, dtype=grid.x_min.dtype)
    halfspan = 0.5 * grid.size_m - 0.5 * grid.res
    x_min_new_all = cx - halfspan
    y_min_new_all = cy - halfspan
    for b in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        fb = grid.h_bias if fill_world is None else float(fill_world[b].item())
        sx = int(((grid.x_min[b] - x_min_new_all[b]) / res).round().to(torch.int64).item())
        sy = int(((grid.y_min[b] - y_min_new_all[b]) / res).round().to(torch.int64).item())
        grid.x_min[b] = grid.x_min[b] - sx * grid.res
        grid.y_min[b] = grid.y_min[b] - sy * grid.res
        if sx == 0 and sy == 0:
            continue
        m = torch.roll(grid.mean[b:b + 1], shifts=(sy, sx), dims=(-2, -1))
        v = torch.roll(grid.var[b:b + 1], shifts=(sy, sx), dims=(-2, -1))
        if sx > 0:
            m[:, :, :, :sx] = fb; v[:, :, :, :sx] = grid.init_var
        elif sx < 0:
            m[:, :, :, sx:] = fb; v[:, :, :, sx:] = grid.init_var
        if sy > 0:
            m[:, :, :sy, :] = fb; v[:, :, :sy, :] = grid.init_var
        elif sy < 0:
            m[:, :, sy:, :] = fb; v[:, :, sy:, :] = grid.init_var
        grid.mean[b:b + 1] = m
        grid.var[b:b + 1] = v


def test_vectorized_recenter_matches_loop():
    torch.manual_seed(0)
    for fill in (None, "per_env"):
        B = 7
        a = GlobalGrid(batch_size=B, size_m=3.0, res=0.1, init_var=10.0, h_bias=-3.0)
        b = GlobalGrid(batch_size=B, size_m=3.0, res=0.1, init_var=10.0, h_bias=-3.0)
        var0 = torch.rand(B, 1, 30, 30)
        for g in (a, b):
            g.mean.copy_(torch.arange(B * 30 * 30, dtype=torch.float32).view(B, 1, 30, 30))
            g.var.copy_(var0)
        for _ in range(5):
            mask = torch.rand(B) < 0.7
            cx = torch.randn(B) * 1.5  # includes shifts larger than the map
            cy = torch.randn(B) * 1.5
            fw = torch.randn(B) if fill else None
            a.recenter(mask, cx.clone(), cy.clone(), fill_world=fw)
            _reference_recenter(b, mask, cx.clone(), cy.clone(), fill_world=fw)
            assert torch.equal(a.mean, b.mean)
            assert torch.equal(a.var, b.var)
            assert torch.allclose(a.x_min, b.x_min) and torch.allclose(a.y_min, b.y_min)


def _terrain(x, y):
    """World terrain: 0.3 m step at x > 1, plus a 0.5 m box."""
    z = torch.where(x > 1.0, torch.full_like(x, 0.3), torch.zeros_like(x))
    box = (x > 2.0) & (x < 2.6) & (y.abs() < 0.3)
    return torch.where(box, torch.full_like(x, 0.8), z)


def _dense_scan(origin, yaw, spec, step=0.02):
    """Dense top-down 'point cloud' around each origin (placeholder ray caster)."""
    xs = torch.arange(-3.3, 3.3, step)
    ys = torch.arange(-1.8, 1.8, step)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    n = origin.shape[0]
    c, s = torch.cos(yaw).view(n, 1), torch.sin(yaw).view(n, 1)
    lx, ly = gx.reshape(1, -1), gy.reshape(1, -1)
    wx = c * lx - s * ly + origin[:, 0:1]
    wy = s * lx + c * ly + origin[:, 1:2]
    return torch.stack([wx, wy, _terrain(wx, wy)], -1)


def test_dummy_pipeline_fuses_and_queries_ground_truth():
    torch.manual_seed(0)
    spec = LocalGridSpec()
    Xs, Ys = spec.axes()
    B = 3
    pipe = LivoxGlobalProbWinnerPipeline(Xs, Ys, batch_size=B, res=spec.res, size_m=6.0, init_var=10.0, tau_min=0.5)
    model = DummyMappingModel(0.01)
    sensor_h = 1.2
    pos = torch.tensor([[-1.0, 0.0], [-1.0, 0.2], [-0.5, -0.1]])
    yaw = torch.tensor([0.0, 0.2, -0.3])
    pipe.seed_episode(torch.arange(B), pos[:, 0], pos[:, 1], torch.full((B,), sensor_h), yaw)
    last = torch.cat([pos, torch.full((B, 1), sensor_h), yaw.unsqueeze(1)], 1)
    # drive forward 2.5 m in 10 frames -> triggers auto-recentering (margin 0.3 * 6 m)
    for k in range(10):
        pos = pos + torch.stack([torch.cos(yaw), torch.sin(yaw)], 1) * 0.25
        origin = torch.cat([pos, torch.full((B, 1), sensor_h)], 1)
        em = points_to_local_grid(_dense_scan(origin, yaw, spec), origin, yaw, spec)
        mu, logv = model(em.unsqueeze(1))
        cur = torch.cat([origin, yaw.unsqueeze(1)], 1)
        dx, dy = cur[:, 0] - last[:, 0], cur[:, 1] - last[:, 1]
        c, s = torch.cos(-cur[:, 3]), torch.sin(-cur[:, 3])
        pipe.set_pose(last[:, 0], last[:, 1], last[:, 2], last[:, 3])
        pipe.step(c * dx - s * dy, s * dx + c * dy, cur[:, 2] - last[:, 2], cur[:, 3] - last[:, 3], mu[:, 0], logv[:, 0])
        last = cur
    assert torch.allclose(pipe.x, pos[:, 0], atol=1e-5) and torch.allclose(pipe.y, pos[:, 1], atol=1e-5)

    # query at a base pose 0.5 m below the sensor (export grid like the sensor's 26 x 15 @ 8 cm)
    base_z = sensor_h - 0.5
    pipe.set_pose(pos[:, 0], pos[:, 1], torch.full((B,), base_z), yaw)
    xs_b = (torch.arange(26) - 12.5) * 0.08 + 0.5
    ys_b = (torch.arange(15) - 7.0) * 0.08
    Xb, Yb = torch.meshgrid(xs_b, ys_b, indexing="ij")
    mean, var = pipe.export_to_base(Xb, Yb)
    c, s = torch.cos(yaw).view(B, 1, 1), torch.sin(yaw).view(B, 1, 1)
    wx = c * Xb - s * Yb + pos[:, 0].view(B, 1, 1)
    wy = s * Xb + c * Yb + pos[:, 1].view(B, 1, 1)
    gt = _terrain(wx, wy) - base_z
    err = (mean[:, 0] - gt).abs()
    # nearest-cell export: errors only at height discontinuities (within a cell of an edge)
    assert (err < 1e-4).float().mean() > 0.9
    # Probabilistic winner-take-all (paper Eq. 6): R_eff = max(R, 0.5 * prior), so the variance of a
    # repeatedly observed cell halves per frame from init_var (10) towards the dummy 0.01 -- never below it
    assert (var >= 0.01 - 1e-6).all()
    assert (var < 10.0).float().mean() > 0.9
    assert (var < 1.0).float().mean() > 0.5


def test_probabilistic_winner_rules():
    spec = LocalGridSpec(x_range=(-0.3, 0.3), y_range=(-0.3, 0.3), res=0.1)
    Xs, Ys = spec.axes()
    # odd cell count (21) -> global cell centres coincide with the local ones at the spawn pose
    pipe = LivoxGlobalProbWinnerPipeline(Xs, Ys, batch_size=1, res=0.1, size_m=2.1, init_var=10.0, tau_min=0.5)
    pipe.seed_episode(torch.tensor([0]), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    zero = torch.zeros(1)
    L, W = spec.shape
    # confident measurement over an unseen map: R_eff = max(1e-4, 0.5 * 10) = 5 < 1.5 * 10 -> valid,
    # p_win = (1/5) / (1/5 + 1/10) = 2/3 per frame; repeated frames converge to the measurement
    for _ in range(30):
        pipe.step(zero, zero, zero, zero, torch.full((1, L, W), -1.0), torch.full((1, L, W), math.log(1e-4)))
    m, v = pipe.export_global()
    seen = v[0, 0] < 10.0
    assert seen.sum() == L * W
    assert torch.allclose(m[0, 0][seen], torch.tensor(-1.0))
    # variance is lower-bounded by tau_min * prior each time -> shrinks geometrically, never below the measurement
    assert (v[0, 0][seen] >= 1e-4 - 1e-9).all()
    v_conf = v.clone()
    # an uncertain contradicting measurement (var 1.0 >> 1.5 * prior and > 0.2^2) never wins
    for _ in range(20):
        pipe.step(zero, zero, zero, zero, torch.full((1, L, W), 0.5), torch.zeros(1, L, W))
    m2, v2 = pipe.export_global()
    assert torch.allclose(m2[0, 0][seen], torch.tensor(-1.0))
    assert torch.equal(v2, v_conf)
    # hole predictions never update the map
    pipe.step(zero, zero, zero, zero, torch.full((1, L, W), LIVOX_HOLE), torch.full((1, L, W), -9.0))
    assert torch.equal(pipe.export_global()[0], m2)


def test_partial_env_updates_leave_other_envs_untouched():
    spec = LocalGridSpec(x_range=(-0.3, 0.3), y_range=(-0.3, 0.3), res=0.1)
    Xs, Ys = spec.axes()
    pipe = LivoxGlobalProbWinnerPipeline(Xs, Ys, batch_size=4, res=0.1, size_m=2.0)
    pipe.seed_episode(torch.arange(4), torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.zeros(4))
    before = pipe.grid.mean.clone()
    ids = torch.tensor([1, 3])
    L, W = spec.shape
    z = torch.zeros(2)
    pipe.step(z, z, z, z, torch.full((2, L, W), -1.0), torch.full((2, L, W), math.log(1e-4)), ids=ids)
    assert torch.equal(pipe.grid.mean[[0, 2]], before[[0, 2]])
    assert not torch.equal(pipe.grid.mean[[1, 3]], before[[1, 3]])
