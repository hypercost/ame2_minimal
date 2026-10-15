"""Layer 2: targeted unit scenes for the mapping model (Warp ray tracing, no Isaac / Kit).

Hand-built meshes, Mid360 at 1.2 m above the ground looking along +x:

* ``box``      -- 0.5 m box ahead; check: the shadow behind it gets high sigma;
* ``stair``    -- single 0.17 m step edge; check: max height step across the edge is kept;
* ``stones``   -- sparse 0.4 m stepping stones over a 1 m deep pit; check: gaps are not filled
                  (predicted height clearly below the stone top, or high sigma);
* ``floating`` -- box hovering 0.9-1.1 m above the ground (report only).

Each scene: PNG with input / prediction / sigma / gate / GT / |error| side by side, plus
``checks.json``.

    python ame2/scripts/mapping/eval/unit_scenes.py --checkpoint logs/mapping/mapping_ckpt.pt \\
        --out_dir logs/mapping_eval/scenes
"""

from __future__ import annotations

import argparse
import json
import math
import os

import _common
import numpy as np
import torch

from ame2.mapping import terrain_meshes as tm
from ame2.mapping.augment import AugmentCfg
from ame2.mapping.eval_metrics import check_edge_sharpness, check_gaps_not_filled, check_shadow_sigma
from ame2.mapping.grid import LocalGridSpec
from ame2.mapping.warp_sampler import SamplerCfg, WarpMappingSampler

SENSOR_H = 1.2


def _mesh(name, *parts):
    v, f = tm.merge(*parts)
    return tm.TerrainMesh(name, v, f, (-10, 10, -10, 10), probe_z=5.0)


def build_scenes():
    plane = tm.plane_mesh(-15, 15, -15, 15, 0.0)
    one = lambda a: np.array([a], dtype=np.float32)  # noqa: E731
    box = _mesh("box", plane, tm.boxes_mesh(one([1.3, 0.0]), one([0.6, 1.2]), one(0.0), one(0.5), one(0.0)))
    stair = _mesh("stair", plane, tm.boxes_mesh(one([6.0, 0.0]), one([10.0, 20.0]), one(0.0), one(0.17), one(0.0)))
    # stones: tops at 0, pit floor at -1; 0.4 m stones on a 0.7 m pitch -> 0.3 m gaps
    floor = tm.plane_mesh(-15, 15, -15, 15, -1.0)
    c = np.stack(np.meshgrid(np.arange(-5, 5.01, 0.7), np.arange(-3, 3.01, 0.7), indexing="ij"), -1).reshape(-1, 2)
    n = c.shape[0]
    stones = _mesh("stones", floor, tm.boxes_mesh(c.astype(np.float32), np.full((n, 2), 0.4, np.float32),
                                                  np.full(n, -1.0, np.float32), np.zeros(n, np.float32),
                                                  np.zeros(n, np.float32)))
    floating = _mesh("floating", plane, tm.boxes_mesh(one([1.5, 0.0]), one([1.0, 1.0]), one(0.9), one(1.1), one(0.0)))
    return {"box": (box, 0.0), "stair": (stair, 0.0), "stones": (stones, 0.0), "floating": (floating, 0.0)}


def run_checks(name, spec, x, y, mu, sigma):
    X, Y = spec.axes()
    obs = x > spec.hole_value + 0.1
    ground = (y - (-SENSOR_H)).abs() < 0.02
    if name == "box":
        shadow = ground & ~obs & (X > 1.7) & (X < 2.6) & (Y.abs() < 0.4)
        visible = ground & obs & (X > 0.3) & (X < 0.9) & (Y.abs() < 0.5)
        if not shadow.any() or not visible.any():
            return {"pass": False, "error": "empty shadow/visible region"}
        return check_shadow_sigma(sigma, shadow, visible)
    if name == "stair":
        band = (Y.abs() < 0.5) & (X > 0.7) & (X < 1.3)
        return check_edge_sharpness(mu, y, band, tol=0.3)
    if name == "stones":
        top = -SENSOR_H
        gap = (y < top - 0.5) & (X.abs() < 2.5) & (Y.abs() < 1.2)
        stone = ((y - top).abs() < 0.02) & (X.abs() < 2.5) & (Y.abs() < 1.2)
        return check_gaps_not_filled(mu, sigma, gap, stone_top=top, stone_mask=stone)
    obs_err = float((mu - y).abs()[obs].mean()) if obs.any() else float("nan")
    region = (X > 0.9) & (X < 2.1) & (Y.abs() < 0.6)
    return {"report_only": True, "mae_under_box_region": float((mu - y).abs()[region].mean()),
            "observed_mae": obs_err, "gt_under_box": float(y[region].mean()), "pred_under_box": float(mu[region].mean())}


def render_png(path, spec, title, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [spec.y_range[1], spec.y_range[0], spec.x_range[0], spec.x_range[1]]   # x up, y left
    fig, axs = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 5.2))
    for ax, (lab, img, kw) in zip(axs, panels):
        if img is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center"), ax.set_axis_off(), ax.set_title(lab)
            continue
        im = ax.imshow(np.flipud(np.asarray(img)), extent=ext, aspect="equal", **kw)
        ax.set_title(lab, fontsize=9), ax.set_xlabel("y [m]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axs[0].set_ylabel("x (forward) [m]")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="ckpt / TorchScript path or 'dummy'")
    p.add_argument("--rays_per_frame", type=int, default=20000, help="one Mid360 window (as in sim)")
    p.add_argument("--augment", action="store_true", help="apply the training grid augmentations to the input")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="logs/mapping_eval/scenes")
    p.add_argument("--x_range", type=float, nargs=2, default=(-3.0, 3.0))
    p.add_argument("--y_range", type=float, nargs=2, default=(-1.5, 1.5))
    p.add_argument("--res", type=float, default=0.03)
    a = p.parse_args(argv)
    spec = LocalGridSpec(tuple(a.x_range), tuple(a.y_range), a.res)
    device = torch.device(a.device)
    model = _common.load_model(a.checkpoint, device)
    os.makedirs(a.out_dir, exist_ok=True)
    scenes = build_scenes()
    cfg = SamplerCfg(rays_per_frame=a.rays_per_frame, mid360_prob=1.0, sensor_xy_jitter=0.0, sensor_z_jitter=0.0,
                     max_tilt_deg=0.0, aug=AugmentCfg())
    sampler = WarpMappingSampler(spec, cfg, device=str(device), seed=a.seed, meshes=[m for m, _ in scenes.values()])
    results = {}
    for mi, (name, (mesh, ground)) in enumerate(scenes.items()):
        pos = torch.tensor([[0.0, 0.0, ground + SENSOR_H if name != "stones" else SENSOR_H]], device=device)
        x, y, ok = sampler.render(mi, pos, torch.zeros(1, device=device), clean=not a.augment)
        mu, sigma, gate = _common.predict(model, x.unsqueeze(1))
        x, y, mu, sigma = x[0].cpu(), y[0].cpu(), mu[0, 0].cpu(), sigma[0, 0].cpu()
        gate = None if gate is None else gate[0, 0].cpu()
        chk = run_checks(name, spec, x, y, mu, sigma)
        chk["observed_fraction"] = float((x > spec.hole_value + 0.1).float().mean())
        results[name] = chk
        hv = dict(cmap="terrain", vmin=-2.4, vmax=0.2)
        xin = np.where(x.numpy() > spec.hole_value + 0.1, x.numpy(), np.nan)
        status = "report only" if chk.get("report_only") else ("PASS" if chk.get("pass") else "FAIL")
        render_png(os.path.join(a.out_dir, f"scene_{name}.png"), spec,
                   f"{name}: {status}  ({_common.model_name(a.checkpoint)})",
                   [("input (holes blank)", xin, hv), ("prediction", mu, hv),
                    ("sigma [m]", sigma, dict(cmap="magma", vmin=0, vmax=max(0.3, float(sigma.max())))),
                    ("gate", gate, dict(cmap="viridis", vmin=0, vmax=1)), ("ground truth", y, hv),
                    ("|error| [m]", (mu - y).abs(), dict(cmap="Reds", vmin=0, vmax=0.5))])
        print(f"[scene] {name}: {status} {json.dumps({k: v for k, v in chk.items() if k != 'pass'}, default=float)}")
    passed = [n for n, r in results.items() if not r.get("report_only")]
    summary = {"checkpoint": a.checkpoint, "scenes": results,
               "all_checks_pass": all(results[n].get("pass") for n in passed)}
    with open(os.path.join(a.out_dir, "checks.json"), "w") as f:
        json.dump(summary, f, indent=1, default=lambda v: None if (isinstance(v, float) and math.isnan(v)) else float(v))
    print(f"[scene] wrote {a.out_dir}/checks.json and scene_*.png; all checks pass: {summary['all_checks_pass']}")
    return summary


if __name__ == "__main__":
    main()
