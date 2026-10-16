"""Layer 4 closed-loop eval (teacher/student, map mode A/B/C) on IsaacLab 3.0 (kitless OVPhysX).

Same logic and outputs as ``scripts/mapping/eval/closed_loop_abc.py`` (2.x / Kit), but launched
through ``isaaclab.app.launch_simulation`` with Hydra preset selection (``physics=ovphysx``).

  PYTHONPATH=$PWD/ame2:$PWD/rsl_rl /home/zj49/workspace/IS3/.venv/bin/python \
      ame2/scripts/is3/closed_loop_abc.py --map_mode A --num_envs 64 --episodes_per_env 1 \
      --out logs/is3/eval/teacher_A.json physics=ovphysx
"""

import argparse
import json
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "..", "mapping", "eval")
NN = os.path.join(HERE, "..", "..", "ame2", "NN_models")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Ame2-G1-Gaze-Student")
parser.add_argument("--map_mode", choices=("A", "B", "C"), default="A")
parser.add_argument("--policy", choices=("teacher", "student"), default="teacher")
parser.add_argument("--teacher_jit", default=os.path.join(NN, "g1_gaze_teacher_isaaclab.jit"))
parser.add_argument("--student_jit", default=None)
parser.add_argument("--mapping_model", default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes_per_env", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--terrain", default=None)
parser.add_argument("--difficulty", type=float, default=None)
parser.add_argument("--success_dist", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
add_launcher_args(parser)

import ame2.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

args, rest = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + rest
env_cfg, _ = resolve_task_config(args.task, None)

with launch_simulation(env_cfg, args):
    import torch

    sys.path.insert(0, os.path.abspath(EVAL_DIR))
    from _isaac_env import env_terrain_names, load_policy, make_eval_env

    from ame2.mapping.eval_metrics import EpisodeOutcomeAggregator, check_map_compat, policy_map_input

    a = args
    out = a.out or f"logs/is3/eval/{a.policy}_{a.map_mode}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    env, env_cfg, mode = make_eval_env(a.task, a.num_envs, a.device, a.seed, a.map_mode, a.mapping_model,
                                       a.terrain, a.difficulty, env_cfg=env_cfg)
    uenv = env.unwrapped
    torch.manual_seed(a.seed)
    path = a.teacher_jit if a.policy == "teacher" else a.student_jit
    policy = load_policy(a.policy, os.path.abspath(path) if path else path, uenv.device)
    agg = EpisodeOutcomeAggregator(env_terrain_names(uenv), a.success_dist, a.episodes_per_env)

    obs, _ = env.reset()
    compat = check_map_compat(obs["neural_map"], obs["teacher_mapping"])
    print(f"[layer4] map obs compatibility: {compat}", flush=True)
    steps = 0
    with torch.inference_mode():
        while steps < a.max_steps and not agg.finished():
            dist = torch.norm(uenv.command_manager.get_command("base_pose")[:, :2], dim=1)
            actions = policy(obs, policy_map_input(a.map_mode, obs, a.policy))
            obs, _, term, trunc, _ = env.step(actions)
            agg.update(term | trunc, trunc & ~term, dist)
            steps += 1
            if steps % 500 == 0:
                o = agg.summary()["overall"]
                print(f"[layer4 {a.map_mode} {steps}] episodes {o['episodes']} success {o['success_rate']:.3f} "
                      f"fall {o['fall_rate']:.3f}", flush=True)
    res = {"backend": "isaaclab3", "map_mode": a.map_mode, "mode": mode, "policy": a.policy, "policy_path": path,
           "mapping_model": a.mapping_model, "task": a.task, "seed": a.seed, "steps": steps,
           "compat": compat, "outcomes": agg.summary()}
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"[layer4] wrote {out}: {res['outcomes']['overall']}", flush=True)
    env.close()
