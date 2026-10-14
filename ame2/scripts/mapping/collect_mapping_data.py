"""[LEGACY / OPTIONAL] Record (Mid360 grid, GT grid) pairs by playing the teacher in Isaac Lab.

Not the recommended data path any more: the paper (Sec. V-B1) samples frames with
Warp ray tracing from random poses, without physics -- see
``ame2/mapping/warp_sampler.py`` and ``train_mapping_model.py --source warp``.
Keep this script for optional fine-tuning on the in-sim sensor pipeline
(``train_mapping_model.py --source shards --data_dir ...``).

Each sample is captured at one robot pose:

* input  ``x``: the Mid360 scan scattered into the model grid
  (:func:`ame2.mapping.grid.points_to_local_grid`; holes = -3) -- exactly what
  :class:`ame2.sensors.LivoxNeuralMapSensor` feeds its mapping model;
* label  ``y``: the dense top-down ray caster scattered into the same grid
  (same origin = Mid360 position, same base-yaw axes), pin-holes closed.

Shards are written as ``torch.save({"x": (N,L,W) fp16, "y": (N,L,W) fp16, "spec": {...}})``
and consumed by ``train_mapping_model.py --data_dir``.

Usage::

    python ame2/scripts/export_teacher_jit.py --checkpoint modelzoo/g1_gaze_64000env_10h/model_2400.pt
    python ame2/scripts/mapping/collect_mapping_data.py --headless --num_envs 256 \\
        --num_samples 200000 --out_dir data/mapping
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect neural-mapping training data with the teacher in the loop.")
parser.add_argument("--task", type=str, default="Ame2-G1-Mapping-Collect")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument(
    "--teacher_jit", type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "..", "ame2", "NN_models", "g1_gaze_teacher_isaaclab.jit"),
    help="TorchScript teacher (see ame2/scripts/export_teacher_jit.py)",
)
parser.add_argument("--num_samples", type=int, default=100000)
parser.add_argument("--record_every", type=int, default=5, help="record every N policy steps (5 -> 10 Hz at 50 Hz)")
parser.add_argument("--envs_per_record", type=int, default=64, help="random envs sampled per record step")
parser.add_argument("--warmup_steps", type=int, default=50, help="policy steps before recording starts")
parser.add_argument("--shard_size", type=int, default=4096)
parser.add_argument("--out_dir", type=str, default="data/mapping")
parser.add_argument("--base_body", type=str, default="base", help="body whose yaw defines the grid axes")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ame2.tasks  # noqa: F401  (registers the gym ids)
from ame2.mapping.grid import LocalGridSpec, fill_isolated_holes, points_to_local_grid
from ame2.sensors.neural_mapping_utils import LIVOX_HOLE


def _yaw_from_q(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    device = uenv.device

    teacher_path = os.path.abspath(args_cli.teacher_jit)
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(f"Teacher JIT not found: {teacher_path} (run ame2/scripts/export_teacher_jit.py)")
    teacher = torch.jit.load(teacher_path, map_location=device).eval()

    # same grid as LivoxNeuralMapSensorCfg defaults (201 x 101 @ 3 cm, +-3 m x +-1.5 m)
    spec = LocalGridSpec(x_range=(-3.0, 3.0), y_range=(-1.5, 1.5), res=0.03, hole_value=LIVOX_HOLE, z_max=2.0)
    robot = uenv.scene["robot"]
    base_idx = robot.find_bodies(args_cli.base_body)[0][0]
    lidar = uenv.scene.sensors["lidar_cam"]
    gt_scanner = uenv.scene.sensors["mapping_gt_scanner"]

    os.makedirs(args_cli.out_dir, exist_ok=True)
    buf_x, buf_y, n_buf, n_total, shard_id = [], [], 0, 0, 0

    def flush():
        nonlocal buf_x, buf_y, n_buf, shard_id
        if n_buf == 0:
            return
        path = os.path.join(args_cli.out_dir, f"shard_{shard_id:05d}.pt")
        torch.save(
            {
                "x": torch.cat(buf_x).cpu(),
                "y": torch.cat(buf_y).cpu(),
                "spec": {"x_range": spec.x_range, "y_range": spec.y_range, "res": spec.res,
                         "hole_value": spec.hole_value, "z_max": spec.z_max},
            },
            path,
        )
        print(f"[INFO] wrote {path} ({n_buf} samples, total {n_total})")
        buf_x, buf_y, n_buf = [], [], 0
        shard_id += 1

    obs, _ = env.reset()
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running() and n_total < args_cli.num_samples:
            actions = teacher(obs["teacher_prop"], obs["teacher_mapping"])[0]
            obs, _, _, _, _ = env.step(actions)
            step += 1
            if step < args_cli.warmup_steps or step % args_cli.record_every != 0:
                continue

            n_env = uenv.num_envs
            k = min(args_cli.envs_per_record, n_env)
            ids = torch.randperm(n_env, device=device)[:k]
            origin = lidar.data.pos_w[ids]
            yaw = _yaw_from_q(robot.data.body_quat_w[ids, base_idx])
            x = points_to_local_grid(lidar.data.ray_hits_w[ids], origin, yaw, spec)
            y = points_to_local_grid(gt_scanner.data.ray_hits_w[ids], origin, yaw, spec)
            y = fill_isolated_holes(y, spec.hole_value)
            # drop samples whose label is mostly missing (e.g. off the terrain border)
            ok = (y > spec.hole_value + 0.1).float().mean(dim=(1, 2)) > 0.9
            if ok.any():
                buf_x.append(x[ok].half())
                buf_y.append(y[ok].half())
                n = int(ok.sum())
                n_buf += n
                n_total += n
            if n_buf >= args_cli.shard_size:
                flush()
    flush()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
