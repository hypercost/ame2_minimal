"""Layer 3: in-loop fused map vs. GT height scan in Isaac Lab (UNRUN on the dev box: needs Kit + GPU).

The teacher JIT drives the robot on the GT map; every policy step the student's map
observation (``neural_map``: fused global map queried at the base pose, 26 x 15 x 4) is
compared with the GT teacher height scan at the same cells (``teacher_mapping``,
26 x 15 x 3), after checking that both grids match. Logs MAE (all GT-valid cells and
observed cells only), coverage, frame-to-frame jumps of the error, per terrain / per env,
and straight vs. yaw-turn statistics. Outputs: steps.csv, per_env.csv, summary.json, plots.

    python ame2/scripts/mapping/eval/isaac_map_vs_gt.py --headless --num_envs 256 --steps 3000 \\
        --mapping_model ame2/ame2/NN_models/g1LivoxMapping_perframe.pt --out_dir logs/mapping_eval/inloop_B
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Ame2-G1-Gaze-Student")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000, help="policy steps (50 Hz)")
parser.add_argument("--warmup_steps", type=int, default=25, help="steps excluded from the statistics")
parser.add_argument("--map_mode", choices=("B", "C"), default="B", help="B learned model, C raw grid + const var")
parser.add_argument("--mapping_model", default=None)
parser.add_argument("--teacher_jit", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "ame2",
                                                          "NN_models", "g1_gaze_teacher_isaaclab.jit"))
parser.add_argument("--terrain", default=None, help="restrict to one sub-terrain")
parser.add_argument("--difficulty", type=float, default=None)
parser.add_argument("--yaw_rate_thr", type=float, default=0.5, help="rad/s; above = 'turning'")
parser.add_argument("--jump_thr", type=float, default=0.05, help="m; frame-to-frame MAE change counted as jump")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--out_dir", default="logs/mapping_eval/inloop")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import csv  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _isaac_env import env_terrain_names, load_policy, make_eval_env  # noqa: E402

from ame2.mapping.eval_metrics import (  # noqa: E402
    STUDENT_CLIP,
    MapErrorTracker,
    check_map_compat,
    neural_to_teacher_map,
)


def plots(tracker, summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = tracker.rows
    t = [r["step"] for r in rows]
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(t, [r["mae_all"] for r in rows], label="MAE (GT-valid cells)")
    axs[0].plot(t, [r["mae_observed"] for r in rows], label="MAE (observed cells)")
    axs[0].set_ylabel("[m]"), axs[0].legend(), axs[0].grid(True)
    axs[1].plot(t, [r["coverage"] for r in rows]), axs[1].set_ylabel("coverage"), axs[1].grid(True)
    axs[2].plot(t, [r["jump"] for r in rows], label="|ΔMAE| per step")
    axs[2].plot(t, [r["turning_frac"] for r in rows], label="fraction of envs turning", alpha=0.6)
    axs[2].set_xlabel("policy step"), axs[2].legend(), axs[2].grid(True)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "map_error_over_time.png"), dpi=110), plt.close(fig)
    pt = summary["per_terrain"]
    names = sorted(pt)
    fig, ax = plt.subplots(figsize=(max(6, len(names)), 4))
    ax.bar(range(len(names)), [pt[n]["mae_all"] for n in names], 0.4, label="MAE")
    ax.bar([i + 0.4 for i in range(len(names))], [pt[n]["mean_jump"] for n in names], 0.4, label="mean jump")
    ax.set_xticks([i + 0.2 for i in range(len(names))], names, rotation=30), ax.legend(), ax.grid(True, axis="y")
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "map_error_per_terrain.png"), dpi=110), plt.close(fig)


def main():
    a = args_cli
    os.makedirs(a.out_dir, exist_ok=True)
    env, env_cfg, mode = make_eval_env(a.task, a.num_envs, a.device, a.seed, a.map_mode, a.mapping_model,
                                       a.terrain, a.difficulty, a.video, os.path.join(a.out_dir, "videos"),
                                       a.video_length)
    uenv = env.unwrapped
    device = uenv.device
    policy = load_policy("teacher", os.path.abspath(a.teacher_jit), device)
    robot = uenv.scene["robot"]
    scanner = uenv.scene.sensors["height_scanner"]
    names = env_terrain_names(uenv)
    tracker = MapErrorTracker(names, a.yaw_rate_thr, a.jump_thr)

    obs, _ = env.reset()
    compat = check_map_compat(obs["neural_map"], obs["teacher_mapping"])
    print(f"[layer3] map obs compatibility: {compat}")
    if not compat["compatible"]:
        raise RuntimeError(f"student and teacher map observations differ: {compat['issues']}")
    L, W = obs["teacher_mapping"].shape[1:3]
    reset = torch.zeros(uenv.num_envs, dtype=torch.bool, device=device)
    with torch.inference_mode():
        for step in range(a.steps):
            actions = policy(obs, obs["teacher_mapping"])            # teacher drives on GT
            obs, _, term, trunc, _ = env.step(actions)
            reset |= term | trunc
            if step < a.warmup_steps:
                continue
            nm = obs["neural_map"]
            pred = neural_to_teacher_map(nm)[..., 2]                  # teacher frame, [-1.5, 1.5]
            gt = obs["teacher_mapping"][..., 2]
            observed = nm[..., 2] > STUDENT_CLIP[0] + 1e-3            # unseen cells sit at the -2 clip
            gt_valid = torch.isfinite(scanner.data.ray_hits_w[..., 2]).view(-1, L, W)
            yaw_rate = robot.data.root_ang_vel_b[:, 2]
            tracker.update(step, pred, gt, observed, yaw_rate, reset=reset, gt_valid=gt_valid)
            reset[:] = False
            if step % 200 == 0:
                r = tracker.rows[-1]
                print(f"[layer3 {step}] MAE {r['mae_all']:.3f} obs {r['mae_observed']:.3f} cov {r['coverage']:.2f}",
                      flush=True)
    summary = tracker.summary()
    summary.update(map_mode=a.map_mode, mode=mode, mapping_model=a.mapping_model, task=a.task, seed=a.seed,
                   steps=a.steps, compat=compat,
                   pass_criteria={"mae_observed_lt_0.05": summary["mae_observed"] < 0.05,
                                  "turning_mae_le_1.5x_straight":
                                      summary["yaw"]["turning_mae"] <= 1.5 * summary["yaw"]["straight_mae"]})
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    with open(os.path.join(a.out_dir, "steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tracker.rows[0]))
        w.writeheader(), w.writerows(tracker.rows)
    with open(os.path.join(a.out_dir, "per_env.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "terrain", "mae_all"])
        for i, (t, m) in enumerate(zip(names, summary["per_env_mae"])):
            w.writerow([i, t, m])
    plots(tracker, summary, a.out_dir)
    print(f"[layer3] wrote {a.out_dir}: MAE {summary['mae_all']:.4f}, observed {summary['mae_observed']:.4f}, "
          f"turning {summary['yaw']['turning_mae']:.4f} vs straight {summary['yaw']['straight_mae']:.4f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
