import math

import torch

from ame2.mapping.grid import LocalGridSpec, fill_isolated_holes, points_to_local_grid

SPEC = LocalGridSpec()


def _reference_build_em(hits_w, sensor_pos_w, base_yaw, spec):
    """Verbatim logic of the original LivoxNeuralMapSensor._build_em_from_lidar."""
    n, R = hits_w.shape[0], hits_w.shape[1]
    rel = hits_w - sensor_pos_w.unsqueeze(1)
    c = torch.cos(base_yaw).unsqueeze(1)
    s = torch.sin(base_yaw).unsqueeze(1)
    lx = c * rel[..., 0] + s * rel[..., 1]
    ly = -s * rel[..., 0] + c * rel[..., 1]
    zr = rel[..., 2]
    x0, x1 = spec.x_range
    y0, y1 = spec.y_range
    L, W = spec.shape
    P = L * W
    valid = (lx >= x0) & (lx <= x1) & (ly >= y0) & (ly <= y1) & torch.isfinite(zr)
    ix = torch.round((lx - x0) / spec.res).long().clamp(0, L - 1)
    iy = torch.round((ly - y0) / spec.res).long().clamp(0, W - 1)
    cell = ix * W + iy
    benv = torch.arange(n).unsqueeze(1).expand(n, R)
    gidx = (benv * P + cell)[valid]
    grid = torch.full((n * P,), float("-inf"))
    grid.scatter_reduce_(0, gidx, zr[valid], reduce="amax", include_self=True)
    grid = grid.view(n, L, W)
    filled = grid > float("-inf")
    return torch.where(filled, grid.clamp(-3.0, 2.0), torch.full_like(grid, -3.0))


def test_shape_and_axes():
    assert SPEC.shape == (201, 101)
    X, Y = SPEC.axes()
    assert X.shape == (201, 101)
    assert torch.isclose(X[0, 0], torch.tensor(-3.0)) and torch.isclose(X[-1, 0], torch.tensor(3.0))
    assert torch.isclose(Y[0, 0], torch.tensor(-1.5)) and torch.isclose(Y[0, -1], torch.tensor(1.5))


def test_single_points_land_in_expected_cells_with_yaw():
    torch.manual_seed(0)
    n = 4
    origin = torch.randn(n, 3) * 5
    yaw = torch.tensor([0.0, 0.5, -2.0, math.pi])
    X, Y = SPEC.axes()
    # pick random target cells, build their world coordinates and check the round trip
    L, W = SPEC.shape
    ii = torch.randint(0, L, (n, 50))
    jj = torch.randint(0, W, (n, 50))
    lx, ly = X[ii, jj], Y[ii, jj]
    z = torch.rand(n, 50) * 2 - 2.5
    c, s = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    pts = torch.stack([c * lx - s * ly, s * lx + c * ly, z], -1) + origin.unsqueeze(1)
    em = points_to_local_grid(pts, origin, yaw, SPEC)
    for b in range(n):
        for k in range(50):
            # several points can share a cell -> grid holds their max
            same = (ii[b] == ii[b, k]) & (jj[b] == jj[b, k])
            assert torch.isclose(em[b, ii[b, k], jj[b, k]], z[b][same].max(), atol=1e-5)
    # everything else is a hole
    filled = (em > -3.0 + 1e-6).sum()
    assert filled <= n * 50


def test_max_reduce_misses_and_out_of_range():
    origin = torch.zeros(1, 3)
    yaw = torch.zeros(1)
    pts = torch.tensor([[[1.0, 0.0, -1.0], [1.004, 0.004, -0.5], [1.0, 0.0, float("inf")],
                         [10.0, 0.0, -1.0], [0.0, 5.0, -1.0], [float("nan"), 0.0, 0.0], [0.0, 0.0, 9.0]]])
    em, valid = points_to_local_grid(pts, origin, yaw, SPEC, return_valid=True)
    i, j = round((1.0 + 3.0) / 0.03), round(1.5 / 0.03)
    assert torch.isclose(em[0, i, j], torch.tensor(-0.5))  # max of the two points
    assert valid.tolist() == [[True, True, False, False, False, False, True]]
    assert torch.isclose(em[0, 100, 50], torch.tensor(2.0))  # clamped to z_max
    assert int((em > -3.0 + 1e-6).sum()) == 2


def test_random_extra_points_flat_ground():
    """Guide: 'add random extra points to verify the local grid projection works'."""
    torch.manual_seed(1)
    n, R = 3, 20000
    origin = torch.tensor([[0.0, 0.0, 1.2], [3.0, -2.0, 1.4], [-5.0, 1.0, 1.0]])
    yaw = torch.tensor([0.3, -1.0, 2.5])
    xy = (torch.rand(n, R, 2) * 2 - 1) * torch.tensor([3.0, 1.5])
    c, s = torch.cos(yaw).view(n, 1), torch.sin(yaw).view(n, 1)
    wx = c * xy[..., 0] - s * xy[..., 1] + origin[:, 0:1]
    wy = s * xy[..., 0] + c * xy[..., 1] + origin[:, 1:2]
    pts = torch.stack([wx, wy, torch.zeros(n, R)], -1)  # ground at world z = 0
    # + random extra points above a 0.5 m box at local x in [1, 1.5]
    box = (xy[..., 0] > 1.0) & (xy[..., 0] < 1.5)
    pts[..., 2] = torch.where(box, torch.full_like(wx, 0.5), pts[..., 2])
    em = points_to_local_grid(pts, origin, yaw, SPEC)
    filled = em > -2.99
    assert filled.float().mean() > 0.5
    expected_ground = -origin[:, 2].view(n, 1, 1)
    X, _ = SPEC.axes()
    on_ground = filled & ((X < 0.95) | (X > 1.55)).unsqueeze(0)
    assert torch.allclose(em[on_ground], expected_ground.expand_as(em)[on_ground])
    on_box = filled & ((X > 1.05) & (X < 1.45)).unsqueeze(0)
    assert torch.allclose(em[on_box], (0.5 + expected_ground).expand_as(em)[on_box])


def test_matches_original_sensor_logic_inside_grid():
    torch.manual_seed(2)
    n, R = 5, 5000
    origin = torch.randn(n, 3)
    yaw = (torch.rand(n) * 2 - 1) * math.pi
    rel = torch.stack([torch.rand(n, R) * 5.9 - 2.95, torch.rand(n, R) * 2.9 - 1.45, torch.rand(n, R) * 3 - 2.5], -1)
    c, s = torch.cos(yaw).view(n, 1), torch.sin(yaw).view(n, 1)
    pts = torch.stack([c * rel[..., 0] - s * rel[..., 1], s * rel[..., 0] + c * rel[..., 1], rel[..., 2]], -1)
    pts = pts + origin.unsqueeze(1)
    pts[:, ::17] = float("inf")
    assert torch.equal(points_to_local_grid(pts, origin, yaw, SPEC), _reference_build_em(pts, origin, yaw, SPEC))


def test_fill_isolated_holes():
    g = torch.full((1, 20, 20), -1.0)
    g[0, 5, 5] = -3.0  # pin-hole
    g[0, 10:16, 10:16] = -3.0  # real occlusion
    out = fill_isolated_holes(g)
    assert out[0, 5, 5] == -1.0
    assert (out[0, 11:15, 11:15] == -3.0).all()
