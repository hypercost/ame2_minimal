"""Sim-gap diagnosis: locate where the in-loop map error is added (needs Kit + GPU).

Per lidar update (subsampled envs) it compares, on the 201x101 model grid and against a
top-down label traced on the *Isaac* terrain mesh with the training label definition
(sensor-relative, start = sensor z + 2.5 m):
  raw  : runtime input grid (Mid360 max-z)       -> observed cells
  pred : per-frame model output on that grid     -> observed / all cells
  warp : WarpMappingSampler.render on the same Isaac mesh + same pose (clean) and model on it
and per policy step, in teacher format (26x15): fused learned map vs teacher GT, overall,
on flat cells, and per terrain. Signed errors are pred - GT.

    python ame2/scripts/mapping/eval/diag_simgap.py --headless --num_envs 128 --steps 600 \\
        --mapping_model logs/mapping_fix_quick/mapping_ckpt.pt --out_dir logs/mapping_eval/simgap/run0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Ame2-G1-Gaze-Student")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup_steps", type=int, default=25)
parser.add_argument("--map_mode", choices=("B", "C"), default="B")
parser.add_argument("--mapping_model", required=True)
parser.add_argument("--lidar_noise", type=int, default=-1, help="-1 keep task cfg, 0 off, 1 on")
parser.add_argument("--frames_per_update", type=int, default=16)
parser.add_argument("--teacher_jit", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "ame2",
                                                          "NN_models", "g1_gaze_teacher_isaaclab.jit"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", default="logs/mapping_eval/simgap/run0")
parser.add_argument("--dump_frames", type=int, default=8, help="save this many raw frames for inspection")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaaclab_tasks.utils as _u  # noqa: E402,F401
from _isaac_env import env_terrain_names, load_policy  # noqa: E402
import _isaac_env  # noqa: E402

from isaaclab.sensors import RayCaster  # noqa: E402
from isaaclab.utils.warp import raycast_mesh  # noqa: E402

from ame2.mapping.eval_metrics import STUDENT_CLIP, neural_to_teacher_map  # noqa: E402
from ame2.mapping.models import load_mapping_model  # noqa: E402
from ame2.mapping.terrain_meshes import TerrainMesh  # noqa: E402
from ame2.mapping.warp_sampler import SamplerCfg, WarpMappingSampler  # noqa: E402


class Acc:
    def __init__(self):
        self.d = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # sum signed, sum abs, sum sq, n

    def add(self, key, err):
        err = err[torch.isfinite(err)]
        if err.numel() == 0:
            return
        a = self.d[key]
        a[0] += float(err.sum()); a[1] += float(err.abs().sum()); a[2] += float((err ** 2).sum()); a[3] += err.numel()

    def table(self):
        return {k: {"signed": v[0] / v[3], "mae": v[1] / v[3], "rmse": (v[2] / v[3]) ** 0.5, "n": v[3]}
                for k, v in sorted(self.d.items()) if v[3] > 0}


def main():
    a = args_cli
    os.makedirs(a.out_dir, exist_ok=True)
    orig_make = _isaac_env.make_eval_env

    # optional lidar-noise override through the env cfg
    from isaaclab_tasks.utils import parse_env_cfg
    if a.lidar_noise >= 0:
        def patched_parse(*args, **kw):
            cfg = parse_env_cfg(*args, **kw)
            cfg.scene.lidar_cam.enable_noise = bool(a.lidar_noise)
            return cfg
        _isaac_env.parse_env_cfg = patched_parse
    env, env_cfg, mode = orig_make(a.task, a.num_envs, a.device, a.seed, a.map_mode,
                                   os.path.abspath(a.mapping_model), None, None, False, None, 0)
    uenv = env.unwrapped
    dev = uenv.device
    policy = load_policy("teacher", os.path.abspath(a.teacher_jit), dev)
    names = env_terrain_names(uenv)
    nm_sensor = uenv.scene.sensors["neural_map"]
    scanner = uenv.scene.sensors["height_scanner"]
    lidar = uenv.scene.sensors["lidar_cam"]
    spec = nm_sensor.grid_spec
    model = load_mapping_model(os.path.abspath(a.mapping_model), dev).eval()
    wmesh = RayCaster.meshes[lidar.cfg.mesh_prim_paths[0]]
    verts = wmesh.points.numpy().astype(np.float32)
    faces = wmesh.indices.numpy().reshape(-1, 3).astype(np.int32)
    tm = TerrainMesh("isaac", verts, faces, (float(verts[:, 0].min()), float(verts[:, 0].max()),
                                              float(verts[:, 1].min()), float(verts[:, 1].max())),
                     probe_z=float(verts[:, 2].max()) + 1.0)
    sampler = WarpMappingSampler(spec, SamplerCfg(), device=str(dev), seed=a.seed, meshes=[tm])
    Xs, Ys = spec.axes(dev)
    print(f"[diag] mode {mode} lidar noise {lidar.cfg.enable_noise} max_dist {lidar.cfg.max_distance} "
          f"rays {lidar.num_rays} mesh V{verts.shape[0]} F{faces.shape[0]}", flush=True)

    stash = []
    orig_build = nm_sensor._build_em_from_lidar

    def build(hits_w, sensor_pos_w, base_yaw, ids=None):
        em = orig_build(hits_w, sensor_pos_w, base_yaw, ids)
        k = min(a.frames_per_update, em.shape[0])
        stash.append((em[:k].clone(), hits_w[:k].clone(), sensor_pos_w[:k].clone(), base_yaw[:k].clone(),
                      ids[:k].clone()))
        return em
    nm_sensor._build_em_from_lidar = build

    def label_on_isaac(pos, yaw):
        k = pos.shape[0]
        c, s = yaw.cos().view(k, 1, 1), yaw.sin().view(k, 1, 1)
        gx = pos[:, 0].view(k, 1, 1) + c * Xs - s * Ys
        gy = pos[:, 1].view(k, 1, 1) + s * Xs + c * Ys
        gz = (pos[:, 2] + 2.5).view(k, 1, 1).expand_as(gx)
        starts = torch.stack([gx, gy, gz], -1).view(k, -1, 3).contiguous()
        dirs = torch.zeros_like(starts); dirs[..., 2] = -1.0
        hits = raycast_mesh(starts, dirs, max_dist=100.0, mesh=wmesh)[0]
        z = hits[..., 2].view(k, *Xs.shape) - pos[:, 2].view(k, 1, 1)
        return z  # inf where no hit

    acc = Acc()
    robot = uenv.scene["robot"]
    dumped = []
    obs, _ = env.reset()
    L, W = obs["teacher_mapping"].shape[1:3]
    with torch.inference_mode():
        for step in range(a.steps):
            stash.clear()
            actions = policy(obs, obs["teacher_mapping"])
            obs, _, term, trunc, _ = env.step(actions)
            if step < a.warmup_steps:
                continue
            # ---- per-frame stages (201x101) ----
            for em, hits, pos, yaw, ids in stash:
                lab = label_on_isaac(pos, yaw)
                ok = torch.isfinite(lab)
                labc = lab.clamp(spec.hole_value, spec.z_max)
                obs_c = (em > spec.hole_value + 1e-3) & ok
                mu, lv = model(em.unsqueeze(1)); mu = mu[:, 0]
                xw, yw, okw = sampler.render(0, pos, yaw, clean=True)
                muw, _ = model(xw.unsqueeze(1)); muw = muw[:, 0]
                obs_w = (xw > spec.hole_value + 1e-3) & okw
                # sampler label vs isaac-raycast label (definition check)
                both = ok & okw
                acc.add("frame/label_warp_minus_isaac", (yw - labc)[both])
                acc.add("frame/raw_obs", (em - labc)[obs_c])
                acc.add("frame/pred_obs", (mu - labc)[obs_c])
                acc.add("frame/pred_all", (mu - labc)[ok])
                acc.add("frame/warp_raw_obs", (xw - yw)[obs_w])
                acc.add("frame/warp_pred_obs", (muw - yw)[obs_w])
                acc.add("frame/coverage_runtime", obs_c.float().mean().view(1))
                acc.add("frame/coverage_warp", obs_w.float().mean().view(1))
                # near the robot (|x|<1.5, |y|<0.6) like the teacher window
                near = ((Xs > -0.6) & (Xs < 1.6) & (Ys.abs() < 0.6)).unsqueeze(0)
                acc.add("frame/raw_obs_near", (em - labc)[obs_c & near])
                acc.add("frame/pred_obs_near", (mu - labc)[obs_c & near])
                acc.add("frame/pred_all_near", (mu - labc)[ok & near])
                for j in range(em.shape[0]):
                    t = names[int(ids[j])]
                    acc.add(f"frame_terrain/{t}/raw_obs", (em[j] - labc[j])[obs_c[j]])
                    acc.add(f"frame_terrain/{t}/pred_obs", (mu[j] - labc[j])[obs_c[j]])
                if len(dumped) < a.dump_frames:
                    dumped.append(dict(em=em[0].cpu(), lab=labc[0].cpu(), mu=mu[0].cpu(), xw=xw[0].cpu(),
                                       yw=yw[0].cpu(), muw=muw[0].cpu(), pos=pos[0].cpu(), yaw=yaw[0].cpu(),
                                       terrain=names[int(ids[0])]))
            # ---- fused, teacher format (26x15) ----
            nm = obs["neural_map"]
            pred = neural_to_teacher_map(nm)[..., 2]
            gt = obs["teacher_mapping"][..., 2]
            observed = nm[..., 2] > STUDENT_CLIP[0] + 1e-3
            gt_valid = torch.isfinite(scanner.data.ray_hits_w[..., 2]).view(-1, L, W)
            m = observed & gt_valid
            unclipped = m & (gt > -1.49) & (gt < 1.49)
            e = pred - gt
            acc.add("fused/obs", e[m])
            acc.add("fused/obs_unclipped", e[unclipped])
            # flat cells: GT 3x3 range < 1 cm
            g = gt.unsqueeze(1)
            rng = (torch.nn.functional.max_pool2d(g, 3, 1, 1) + torch.nn.functional.max_pool2d(-g, 3, 1, 1))[:, 0]
            flat = unclipped & (rng < 0.01)
            acc.add("fused/obs_flat", e[flat])
            acc.add("fused/obs_edge", e[unclipped & (rng > 0.05)])
            for j in range(nm.shape[0]):
                acc.add(f"fused_terrain/{names[j]}/obs", e[j][m[j]])
                acc.add(f"fused_terrain/{names[j]}/flat", e[j][flat[j]])
            if step % 100 == 0:
                t = acc.table()
                print(f"[diag {step}] fused obs {t.get('fused/obs')} frame raw {t.get('frame/raw_obs')} "
                      f"pred {t.get('frame/pred_obs')} warp pred {t.get('frame/warp_pred_obs')}", flush=True)
    t = acc.table()
    with open(os.path.join(a.out_dir, "diag.json"), "w") as f:
        json.dump(dict(table=t, mode=mode, lidar_noise=bool(lidar.cfg.enable_noise), args=vars(a)), f, indent=1)
    torch.save(dumped, os.path.join(a.out_dir, "frames.pt"))
    for k, v in t.items():
        print(f"{k:45s} signed {v['signed']:+.4f} mae {v['mae']:.4f} rmse {v['rmse']:.4f} n {v['n']}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
