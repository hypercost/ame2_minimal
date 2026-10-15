"""Validation tooling: layer-1/2 metrics and scripts (CPU), layer-3/4 pure logic (no Isaac)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ame2.mapping import eval_metrics as em

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(REPO, "ame2", "scripts", "mapping", "eval")


def _load(name):
    sys.path.insert(0, EVAL_DIR)
    spec = importlib.util.spec_from_file_location(name, os.path.join(EVAL_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- layer 1 metrics ----------------

def test_fills_and_distance():
    x = torch.full((1, 1, 9, 9), -3.0)
    x[0, 0, :, :3] = 1.0                       # observed left block
    obs = em.observed_mask(x)
    d = em.distance_to_observed(obs)
    assert d[0, 0, 0, 2] == 0 and d[0, 0, 0, 3] == 1 and d[0, 0, 0, 8] == 6
    mf = em.mean_fill(x, obs)
    assert torch.allclose(mf[~obs], torch.tensor(1.0))
    nf = em.nearest_fill(x, obs)
    assert torch.allclose(nf, torch.ones_like(nf))


def test_calibration_and_sparsification_gaussian():
    g = torch.Generator().manual_seed(0)
    sigma = 0.05 + 0.3 * torch.rand(200_000, generator=g)
    err = (torch.randn(200_000, generator=g) * sigma).abs()
    c = em.calibration(err, sigma)
    assert abs(c["within_1sigma"] - 0.683) < 0.01 and abs(c["within_2sigma"] - 0.954) < 0.01
    good = em.sparsification(err, sigma)
    bad = em.sparsification(err, torch.rand(200_000, generator=g))
    assert 0 <= good["ause"] < bad["ause"]
    assert good["model"][0] == pytest.approx(float(err.mean()), rel=1e-4)
    perfect = em.sparsification(err, err)
    assert perfect["ause"] == pytest.approx(0.0, abs=1e-6)


def test_offline_accumulator_buckets_and_gate():
    torch.manual_seed(0)
    y = torch.zeros(4, 1, 30, 30)
    x = y.clone()
    x[:, :, :, 15:] = -3.0                      # right half unobserved (distances 1..15)
    m = torch.ones_like(x, dtype=torch.bool)
    acc = em.OfflineAccumulator()
    mu = torch.where(x > -2.9, x, torch.full_like(x, 0.1))
    acc.update(x, y, m, mu, torch.full_like(x, 0.1), gate=(x > -2.9).float())
    s = acc.summary()
    assert s["observed_mae"]["model"] == 0 and s["observed_mae"]["raw_input"] == 0
    assert s["hole_mae"]["model"] == pytest.approx(0.1)
    assert s["hole_mae"]["mean_fill"] == pytest.approx(0.0) and s["hole_mae"]["nearest_fill"] == pytest.approx(0.0)
    assert all(s["hole_mae_by_distance"][b]["cells"] > 0 for b in em.BUCKET_NAMES)
    assert s["gate"]["observed_clean"] == 1.0 and s["gate"]["hole"] == 0.0
    assert s["observed_fraction"] == pytest.approx(0.5)


# ---------------- layer 2 checks ----------------

def test_scene_checks_pass_and_fail():
    sig = torch.full((10, 10), 0.05)
    sh = torch.zeros(10, 10, dtype=torch.bool)
    sh[6:] = True
    sig[sh] = 0.4
    assert em.check_shadow_sigma(sig, sh, ~sh)["pass"]
    assert not em.check_shadow_sigma(torch.full((10, 10), 0.05), sh, ~sh)["pass"]
    gt = torch.zeros(10, 4)
    gt[5:] = 0.17
    band = torch.ones(10, 4, dtype=torch.bool)
    assert em.check_edge_sharpness(gt, gt, band)["pass"]
    smooth = torch.linspace(0, 0.17, 10).view(10, 1).expand(10, 4)
    assert not em.check_edge_sharpness(smooth, gt, band)["pass"]
    gap = torch.zeros(5, 5, dtype=torch.bool)
    gap[2] = True
    filled = torch.zeros(5, 5)
    assert not em.check_gaps_not_filled(filled, torch.full((5, 5), 0.05), gap, stone_top=0.0)["pass"]
    assert em.check_gaps_not_filled(filled, torch.full((5, 5), 0.5), gap, stone_top=0.0)["pass"]
    assert em.check_gaps_not_filled(filled - 1.0, torch.full((5, 5), 0.05), gap, stone_top=0.0)["pass"]
    # uncertain everywhere (also on the stones) -> sigma does not excuse filled gaps
    stones = ~gap
    assert not em.check_gaps_not_filled(filled, torch.full((5, 5), 0.5), gap, stone_top=0.0, stone_mask=stones)["pass"]
    sig = torch.full((5, 5), 0.05)
    sig[gap] = 0.5
    assert em.check_gaps_not_filled(filled, sig, gap, stone_top=0.0, stone_mask=stones)["pass"]


# ---------------- layer 3/4 pure logic ----------------

def _maps(B=3, L=26, W=15):
    xs = (torch.arange(L) - (L - 1) / 2) * 0.08 + 0.5
    ys = (torch.arange(W) - (W - 1) / 2) * 0.08
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    z = torch.rand(B, L, W) * 4 - 2
    nm = torch.stack([gx.expand(B, L, W), gy.expand(B, L, W), z, torch.zeros(B, L, W)], -1)
    tm = torch.stack([gx.expand(B, L, W), gy.expand(B, L, W), z.clamp(-1.5, 1.5)], -1)
    return nm, tm


def test_map_compat_and_conversion():
    nm, tm = _maps()
    assert em.check_map_compat(nm, tm)["compatible"]
    t2 = em.neural_to_teacher_map(nm)
    assert t2.shape == tm.shape and torch.allclose(t2, tm)
    assert t2[..., 2].min() >= -1.5 and t2[..., 2].max() <= 1.5
    n2 = em.teacher_to_neural_map(tm)
    assert n2.shape == nm.shape and torch.allclose(n2[..., 3], torch.tensor(em.UNC_CONFIDENT))
    # transposed teacher grid (wrong ordering) must be flagged
    bad = tm.clone()
    bad[..., 0] = bad[..., 0] + 0.08
    r = em.check_map_compat(nm, bad)
    assert not r["compatible"] and "x/y" in r["issues"][0]
    assert not em.check_map_compat(nm[:, :25], tm)["compatible"]


def test_map_modes_and_policy_input():
    for mode, (mt, feed) in {"A": ("dummy", "teacher_mapping"), "B": ("learned", "neural_map"),
                             "C": ("dummy", "neural_map")}.items():
        cfg = SimpleNamespace(model_type="learned", point_source="topdown_gt", dummy_variance=1.0,
                              jit_model_path="x", query_drift_range=0.03, clean_source_ratio=0.1,
                              random_pixel_ratio=0.01)
        info = em.apply_map_mode(cfg, mode, mapping_model="/m.pt")
        assert cfg.model_type == mt and cfg.point_source == "lidar" and info["feed"] == feed
        assert cfg.query_drift_range == 0 and cfg.clean_source_ratio == 0 and cfg.random_pixel_ratio == 0
        assert cfg.dummy_variance == 0.01
        if mode == "B":
            assert cfg.jit_model_path == "/m.pt"
    with pytest.raises(ValueError):
        em.apply_map_mode(SimpleNamespace(), "B")
    with pytest.raises(ValueError):
        em.apply_map_mode(SimpleNamespace(), "D")
    nm, tm = _maps()
    obs = {"neural_map": nm, "teacher_mapping": tm}
    assert em.policy_map_input("A", obs, "teacher") is tm
    assert em.policy_map_input("B", obs, "teacher").shape == tm.shape
    assert em.policy_map_input("C", obs, "student") is nm
    assert em.policy_map_input("A", obs, "student").shape == nm.shape


def test_terrain_column_names_matches_curriculum_rule():
    subs = {"rough": SimpleNamespace(proportion=0.25), "gap": SimpleNamespace(proportion=0.25),
            "stones": SimpleNamespace(proportion=0.5)}
    names = em.terrain_column_names(subs, 8)
    assert names == ["rough", "rough", "gap", "gap", "stones", "stones", "stones", "stones"]


def test_map_error_tracker():
    tr = em.MapErrorTracker(["stones", "stones", "gap"], yaw_rate_thr=0.5, jump_thr=0.05)
    gt = torch.zeros(3, 4, 4)
    pred = gt.clone()
    pred[2] += 0.2
    obs = torch.ones(3, 4, 4, dtype=torch.bool)
    obs[1] = False
    valid = torch.ones_like(obs)
    tr.update(0, pred, gt, obs, torch.tensor([0.0, 0.0, 1.0]), gt_valid=valid)
    pred2 = pred.clone()
    pred2[0] += 0.1
    tr.update(1, pred2, gt, obs, torch.tensor([1.0, 0.0, 1.0]), reset=torch.tensor([False, True, False]))
    s = tr.summary()
    assert s["per_terrain"]["gap"]["mae_all"] == pytest.approx(0.2)
    assert s["per_terrain"]["stones"]["mae_observed"] == pytest.approx(0.05)    # env1 never observed
    assert s["per_terrain"]["stones"]["coverage"] == pytest.approx(0.5)
    assert s["per_terrain"]["stones"]["mean_jump"] == pytest.approx(0.1)        # env1 jump masked (reset)
    assert s["per_terrain"]["stones"]["jump_rate"] == pytest.approx(1.0)
    assert s["yaw"]["turning_mae"] == pytest.approx((0.2 + 0.1 + 0.2) / 3)
    assert len(tr.rows) == 2 and s["per_env_mae"][2] == pytest.approx(0.2)


def test_episode_outcomes_and_compare(tmp_path):
    agg = em.EpisodeOutcomeAggregator(["stones", "stones", "rough"], success_dist=0.5, max_episodes_per_env=1)
    agg.update(torch.tensor([True, True, False]), torch.tensor([True, False, False]), torch.tensor([0.2, 3.0, 0.0]))
    assert not agg.finished()
    agg.update(torch.tensor([True, False, True]), torch.tensor([True, False, True]), torch.tensor([0.1, 0.0, 2.0]))
    assert agg.finished()
    s = agg.summary()
    st = s["per_terrain"]["stones"]
    assert st["episodes"] == 2 and st["success"] == 1 and st["fall"] == 1     # env0's 2nd episode ignored
    assert s["per_terrain"]["rough"]["timeout_far"] == 1
    assert s["focus_terrains"] == ["stones"]
    mod = _load("compare_abc")
    paths = []
    for mode in "ABC":
        p = tmp_path / f"teacher_{mode}.json"
        p.write_text(json.dumps({"map_mode": mode, "outcomes": s}))
        paths.append(str(p))
    table = mod.compare(paths)
    assert "**stones**" in table.splitlines()[2] and "A success" in table
    assert (tmp_path / "abc_table.md").is_file()


def test_isaac_scripts_compile():
    import py_compile

    for f in ("isaac_map_vs_gt.py", "closed_loop_abc.py", "_isaac_env.py", "compare_abc.py"):
        py_compile.compile(os.path.join(EVAL_DIR, f), doraise=True)


# ---------------- layer 1/2 scripts end to end (tiny, CPU) ----------------

def test_layer1_and_layer2_scripts_end_to_end(tmp_path):
    pytest.importorskip("warp")
    pytest.importorskip("matplotlib")
    from ame2.mapping.models import GatedResidualUNet, save_mapping_checkpoint

    ckpt = tmp_path / "ckpt.pt"
    save_mapping_checkpoint(GatedResidualUNet(), str(ckpt), {"base_channels": 8, "hole_value": -3.0})
    small = ["--x_range", "-1.5", "1.5", "--y_range", "-0.75", "0.75", "--res", "0.05"]
    mk = _load("make_testset")
    ts = mk.main(["--frames", "12", "--chunk", "4", "--rays_per_frame", "3000", "--procedural_size", "12",
                  "--device", "cpu", "--out", str(tmp_path / "ts.pt")] + small)
    data = torch.load(ts, weights_only=False)
    assert data["x"].shape[0] == 12 and data["families"] == ["stacked_boxes", "heightfield", "floating_boxes"]
    assert torch.bincount(data["family"].long()).tolist() == [4, 4, 4]
    ev = _load("eval_offline")
    res = ev.main(["--testset", ts, "--checkpoints", str(ckpt), "dummy", "--device", "cpu",
                   "--batch", "5", "--out_dir", str(tmp_path / "off")])
    assert set(res) == {f"{tmp_path.name}/ckpt.pt", "dummy"}
    r = res[f"{tmp_path.name}/ckpt.pt"]
    assert set(r["splits"]) == {"overall", "stacked_boxes", "heightfield", "floating_boxes"}
    assert "gate" in r["splits"]["overall"] and "gate" not in res["dummy"]["splits"]["overall"]
    for f in ("summary.json", "summary.md", "calibration.png", "sparsification.png", "hole_error_by_size.png"):
        assert (tmp_path / "off" / f).is_file(), f
    sc = _load("unit_scenes")
    out = sc.main(["--checkpoint", str(ckpt), "--device", "cpu", "--rays_per_frame", "20000",
                   "--out_dir", str(tmp_path / "sc")])
    assert set(out["scenes"]) == {"box", "stair", "stones", "floating"}
    assert out["scenes"]["floating"]["report_only"]
    for n in ("box", "stair", "stones", "floating"):
        assert (tmp_path / "sc" / f"scene_{n}.png").is_file()
    assert json.loads((tmp_path / "sc" / "checks.json").read_text())["scenes"]["stair"]["gt_max_step"] > 0.15
