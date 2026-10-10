import math
import os

import pytest
import torch

from ame2.mapping.augment import AugmentCfg, augment_inputs, make_synthetic_batch, simulate_occlusion, synthetic_terrain_batch
from ame2.mapping.grid import LocalGridSpec
from ame2.mapping.losses import beta_nll_loss, mapping_loss, total_variation, tv_batch_weights
from ame2.mapping.models import (
    DummyMappingModel,
    GatedResidualUNet,
    export_mapping_jit,
    load_mapping_model,
    save_mapping_checkpoint,
)
from ame2.mapping.pointcloud_noise import apply_pointcloud_noise
from ame2.sensors.neural_mapping_utils import LIVOX_HOLE
import ame2.mapping as mapping_pkg

SMALL = LocalGridSpec(x_range=(-1.2, 1.2), y_range=(-0.6, 0.6), res=0.06)  # 41 x 21


def test_hole_constant_in_sync():
    assert mapping_pkg.LIVOX_HOLE == LIVOX_HOLE


@pytest.mark.parametrize("shape", [(2, 1, 201, 101), (3, 1, 41, 21), (1, 1, 30, 30)])
def test_unet_forward_backward_shapes(shape):
    torch.manual_seed(0)
    model = GatedResidualUNet()
    x = torch.rand(shape) * 2 - 2.0
    x[..., ::3, :] = LIVOX_HOLE
    mu, logv = model(x)
    assert mu.shape == shape and logv.shape == shape
    assert torch.isfinite(mu).all() and torch.isfinite(logv).all()
    assert (logv >= model.logvar_min - 1e-5).all() and (logv <= model.logvar_max + 1e-5).all()
    y = torch.zeros(shape)
    loss, info = mapping_loss(mu, logv, y, torch.ones_like(y, dtype=torch.bool))
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_gate_passes_observed_cells_and_fills_holes():
    model = GatedResidualUNet()
    with torch.no_grad():
        model.head.bias[2] = 50.0  # gate fully open
        model.head.weight.zero_()
    x = torch.full((1, 1, 20, 20), -1.0)
    x[..., 5:10, 5:10] = LIVOX_HOLE
    mu, _ = model(x)
    obs = x > LIVOX_HOLE + 0.1
    assert torch.allclose(mu[obs], x[obs], atol=1e-5)  # observed -> passthrough
    # holes -> raw estimate (= head bias 0 - height_offset), never the -3 sentinel
    assert torch.allclose(mu[~obs], torch.tensor(-model.height_offset))


def test_beta_nll_matches_paper_eq9():
    torch.manual_seed(0)
    mu, logv, y = torch.randn(4, 1, 5, 5), torch.randn(4, 1, 5, 5), torch.randn(4, 1, 5, 5)
    var = logv.exp()
    ref = (var.sqrt() * (0.5 * logv + 0.5 * (y - mu) ** 2 / var)).reshape(4, -1).mean(1)
    assert torch.allclose(beta_nll_loss(mu, logv, y, beta=0.5), ref, atol=1e-6)
    # beta = 0 -> plain Gaussian NLL
    ref0 = (0.5 * logv + 0.5 * (y - mu) ** 2 / var).reshape(4, -1).mean(1)
    assert torch.allclose(beta_nll_loss(mu, logv, y, beta=0.0), ref0, atol=1e-6)
    # stop-gradient on the sigma weight: d/dlogv at the optimum var = (y-mu)^2 is zero
    lv = torch.log(((y - mu) ** 2).clamp_min(1e-3)).requires_grad_(True)
    beta_nll_loss(mu, lv, y, beta=0.5).sum().backward()
    assert lv.grad.abs().max() < 1e-5 or ((y - mu) ** 2 < 1e-3).any()


def test_beta_nll_mask():
    mu = torch.zeros(1, 1, 2, 2)
    y = torch.tensor([[[[0.0, 0.0], [0.0, 100.0]]]])
    m = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.bool)
    assert torch.allclose(beta_nll_loss(mu, torch.zeros_like(mu), y, mask=m), torch.zeros(1))


def test_total_variation_and_weights():
    flat = torch.zeros(1, 1, 10, 10)
    step = torch.zeros(1, 1, 10, 10)
    step[..., 5:, :] = 1.0
    tv = total_variation(torch.cat([flat, step]))
    assert tv[0] == 0 and torch.isclose(tv[1], torch.tensor(10.0 / 100))
    w = tv_batch_weights(tv)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-4) and w[0] == 0
    w2 = tv_batch_weights(tv, floor=1e-3)
    assert w2[0] > 0 and w2[1] > w2[0]


def test_dummy_model_and_loader(tmp_path):
    x = torch.randn(2, 1, 7, 5)
    d = load_mapping_model("dummy", dummy_variance=0.01)
    mu, lv = d(x)
    assert torch.equal(mu, x) and torch.allclose(lv, torch.full_like(x, math.log(0.01)))
    # the dummy model must be scriptable (the sensor may freeze/script models)
    torch.jit.script(DummyMappingModel())


def test_checkpoint_and_jit_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = GatedResidualUNet(base_channels=4).eval()
    x = torch.rand(2, 1, 41, 21) * 2 - 2
    with torch.no_grad():
        ref = model(x)
    ck = str(tmp_path / "ck.pt")
    save_mapping_checkpoint(model, ck, {"base_channels": 4})
    jp = str(tmp_path / "m.jit")
    export_mapping_jit(model, jp)
    for path in (ck, jp):
        m = load_mapping_model(path)
        with torch.no_grad():
            out = m(x)
        assert torch.allclose(out[0], ref[0], atol=1e-5) and torch.allclose(out[1], ref[1], atol=1e-5)
    with pytest.raises(FileNotFoundError):
        load_mapping_model(str(tmp_path / "missing.pt"))


def test_synthetic_terrain_and_occlusion():
    g = torch.Generator().manual_seed(0)
    gt = synthetic_terrain_batch(16, SMALL, generator=g)
    assert gt.shape == (16, 41, 21)
    assert (gt >= LIVOX_HOLE).all() and (gt <= 2.0).all()
    # occlusion: a tall wall in front of the sensor hides the ground behind it
    wall = torch.full((1, 41, 21), -1.2)
    X, _ = SMALL.axes()
    wall[0][(X > 0.5) & (X < 0.62)] = -0.5  # 0.7 m wall, below the sensor
    cfg = AugmentCfg(sensor_xy_jitter=0.0, sensor_z_jitter=0.0, fov_prob=0.0, blind_radius_max=0.0)
    vis = simulate_occlusion(wall, SMALL, cfg)[0]
    assert vis[(X > 0.5) & (X < 0.62)].all()          # wall top visible
    assert not vis[X > 0.9].any()                      # ground behind the wall occluded
    assert vis[(X.abs() < 0.3)].all()                  # ground next to the sensor visible


def test_augment_inputs_and_batch():
    g = torch.Generator().manual_seed(0)
    x, y, m = make_synthetic_batch(8, SMALL, generator=g)
    assert x.shape == y.shape == m.shape == (8, 1, 41, 21)
    assert ((x == LIVOX_HOLE) | ((x > LIVOX_HOLE) & (x <= 2.0))).all()
    holes = (x == LIVOX_HOLE).float().mean()
    assert 0.0 < holes < 1.0
    xc, yc, _ = make_synthetic_batch(8, SMALL, generator=torch.Generator().manual_seed(0), clean=True)
    obs = xc > LIVOX_HOLE
    assert torch.allclose(xc[obs], yc[obs])  # clean = occlusion only, observed cells exact
    a = augment_inputs(yc[:, 0], SMALL, generator=g)
    assert a.shape == (8, 41, 21)


def test_model_learns_on_synthetic_data():
    """A few hundred CPU steps on small grids must reduce the loss clearly."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = GatedResidualUNet(base_channels=8)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    val = make_synthetic_batch(64, SMALL, generator=torch.Generator().manual_seed(99))

    def val_loss():
        with torch.no_grad():
            mu, lv = model(val[0])
            info = mapping_loss(mu, lv, val[1], val[2], tv_reweight=False)[1]
            info["cal2s"] = ((mu - val[1]).abs() < 2 * torch.exp(0.5 * lv)).float().mean()
            return info

    before = val_loss()
    for _ in range(150):
        x, y, m = make_synthetic_batch(32, SMALL, generator=g)
        mu, lv = model(x)
        loss, _ = mapping_loss(mu, lv, y, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = val_loss()
    assert after["beta_nll"] < before["beta_nll"] - 0.05
    # The uncertainty head becomes calibrated quickly; hole-filling MAE only beats a
    # "fill with mean observed height" baseline after ~1k steps, too slow for a unit test.
    assert after["cal2s"] > before["cal2s"]
    assert torch.isfinite(after["mae"])


def test_train_script_end_to_end(tmp_path):
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "ame2", "scripts", "mapping", "train_mapping_model.py")
    spec = importlib.util.spec_from_file_location("train_mapping_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # recorded-data path: write a fake shard in the collector's format
    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    x, y, _ = make_synthetic_batch(40, SMALL, generator=torch.Generator().manual_seed(3))
    torch.save({"x": x[:, 0].half(), "y": y[:, 0].half(),
                "spec": {"x_range": SMALL.x_range, "y_range": SMALL.y_range, "res": SMALL.res,
                         "hole_value": SMALL.hole_value, "z_max": SMALL.z_max}}, shard_dir / "shard_00000.pt")
    out = tmp_path / "out"
    export = tmp_path / "exported.pt"
    hist = mod.main([
        "--data_dir", str(shard_dir), "--synthetic_ratio", "0.5", "--steps", "6", "--clean_steps", "3",
        "--batch_size", "8", "--device", "cpu", "--out_dir", str(out), "--export", str(export),
        "--log_every", "2", "--val_every", "3",
    ])
    assert len(hist) == 6 and all(math.isfinite(h) for h in hist)
    m = load_mapping_model(str(export))
    mu, lv = m(x[:2])
    assert mu.shape == x[:2].shape
    assert load_mapping_model(str(out / "mapping_ckpt.pt")) is not None


def test_pointcloud_noise_ratios():
    torch.manual_seed(0)
    hits = torch.randn(4, 20000, 3)
    hits[:, :1000] = float("inf")
    origin = torch.zeros(4, 3)
    out = apply_pointcloud_noise(hits, origin, drop_ratio=0.15, artifact_ratio=0.02)
    fin_in = torch.isfinite(hits).all(-1)
    fin_out = torch.isfinite(out).all(-1)
    assert not fin_out[:, :1000].any()  # misses stay misses
    drop = (fin_in & ~fin_out).float().sum() / fin_in.float().sum()
    assert abs(drop.item() - 0.15) < 0.01
    moved = fin_out & (out != hits).any(-1)
    assert abs(moved.float().sum().item() / fin_in.float().sum().item() - 0.02 * 0.85) < 0.005
    # artifacts lie on the original ray, closer to the sensor
    r_in = hits[moved].norm(dim=-1)
    r_out = out[moved].norm(dim=-1)
    assert (r_out < r_in).all()
    cos = torch.nn.functional.cosine_similarity(hits[moved], out[moved], dim=-1)
    assert (cos > 0.999).all()
    assert torch.equal(apply_pointcloud_noise(hits, origin, 0.0, 0.0), hits)
