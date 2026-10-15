"""Layer 4: closed-loop A/B/C with the same policy, terrains and seed (UNRUN on the dev box).

One map mode per process (Isaac cannot rebuild a scene in-process reliably); run A, B, C
with the same --seed, then ``--compare`` the JSON files:

  A  GT map            (teacher: ``teacher_mapping``; student: GT converted to the 4-channel format)
  B  learned mapping   (``neural_map`` from the trained model on the Mid360 points)
  C  raw grid + const variance (dummy model on the Mid360 points; holes never fused)

For the teacher in B/C the student map obs is converted with
``eval_metrics.neural_to_teacher_map`` (drop uncertainty, clip z to the teacher's [-1.5, 1.5]).
Success = episode reaches the time limit within 0.5 m of the goal (the goal-reaching
curriculum criterion); fall = any non-time-out termination.

    for M in A B C; do
      python ame2/scripts/mapping/eval/closed_loop_abc.py --headless --map_mode $M --num_envs 512 \\
          --episodes_per_env 2 --mapping_model ame2/ame2/NN_models/g1LivoxMapping_perframe.pt \\
          --out logs/mapping_eval/abc/teacher_$M.json
    done
    python ame2/scripts/mapping/eval/compare_abc.py logs/mapping_eval/abc/teacher_*.json
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--compare", nargs="+", default=None, help="merge result JSONs into a table (no Isaac)")
parser.add_argument("--task", default="Ame2-G1-Gaze-Student")
parser.add_argument("--map_mode", choices=("A", "B", "C"), default="B")
parser.add_argument("--policy", choices=("teacher", "student"), default="teacher")
parser.add_argument("--teacher_jit", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "ame2",
                                                          "NN_models", "g1_gaze_teacher_isaaclab.jit"))
parser.add_argument("--student_jit", default=None, help="student policy JIT (play.py --export)")
parser.add_argument("--mapping_model", default=None)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episodes_per_env", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--terrain", default=None)
parser.add_argument("--difficulty", type=float, default=None)
parser.add_argument("--success_dist", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)


if __name__ == "__main__" and "--compare" in sys.argv:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compare_abc import compare

    compare(parser.parse_known_args()[0].compare)
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _isaac_env import env_terrain_names, load_policy, make_eval_env  # noqa: E402

from ame2.mapping.eval_metrics import EpisodeOutcomeAggregator, check_map_compat, policy_map_input  # noqa: E402


def main():
    a = args_cli
    out = a.out or f"logs/mapping_eval/abc/{a.policy}_{a.map_mode}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    env, env_cfg, mode = make_eval_env(a.task, a.num_envs, a.device, a.seed, a.map_mode, a.mapping_model,
                                       a.terrain, a.difficulty)
    uenv = env.unwrapped
    device = uenv.device
    torch.manual_seed(a.seed)
    path = a.teacher_jit if a.policy == "teacher" else a.student_jit
    policy = load_policy(a.policy, os.path.abspath(path) if path else path, device)
    names = env_terrain_names(uenv)
    agg = EpisodeOutcomeAggregator(names, a.success_dist, a.episodes_per_env)

    obs, _ = env.reset()
    compat = check_map_compat(obs["neural_map"], obs["teacher_mapping"])
    print(f"[layer4] map obs compatibility: {compat}")
    if not compat["compatible"]:
        raise RuntimeError(f"student and teacher map observations differ: {compat['issues']}")
    steps = 0
    with torch.inference_mode():
        while simulation_app.is_running() and steps < a.max_steps and not agg.finished():
            dist = torch.norm(uenv.command_manager.get_command("base_pose")[:, :2], dim=1)
            actions = policy(obs, policy_map_input(a.map_mode, obs, a.policy))
            obs, _, term, trunc, _ = env.step(actions)
            agg.update(term | trunc, trunc & ~term, dist)
            steps += 1
            if steps % 500 == 0:
                o = agg.summary()["overall"]
                print(f"[layer4 {a.map_mode} {steps}] episodes {o['episodes']} success {o['success_rate']:.3f} "
                      f"fall {o['fall_rate']:.3f}", flush=True)
    res = {"map_mode": a.map_mode, "mode": mode, "policy": a.policy, "policy_path": path,
           "mapping_model": a.mapping_model, "task": a.task, "seed": a.seed, "steps": steps,
           "compat": compat, "outcomes": agg.summary()}
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"[layer4] wrote {out}: {res['outcomes']['overall']}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
