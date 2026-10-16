"""IsaacLab 3.0 smoke test: build an AME-2 task and step it with zero/random actions.

Usage (from repo root, IS3 uv venv):
  PYTHONPATH=$PWD/ame2:$PWD/rsl_rl uv run --project /home/zj49/workspace/IS3 --extra ovphysx \
      python ame2/scripts/is3/smoke_env.py --task Ame2-G1-Gaze --num_envs 16 --steps 200 physics=ovphysx
"""

import argparse
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Ame2-G1-Gaze")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--debug", action="store_true")
add_launcher_args(parser)

import ame2.tasks  # noqa: F401,E402  (lazy gym registration)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

args, rest = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + rest

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
env_cfg.scene.num_envs = args.num_envs
print("[smoke] physics cfg:", type(env_cfg.sim.physics).__name__, flush=True)

with launch_simulation(env_cfg, args):
    import gymnasium as gym
    import torch

    t0 = time.time()
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    print(f"[smoke] env created in {time.time()-t0:.1f}s", flush=True)
    if args.debug:
        sc = u.scene if False else env.unwrapped.scene
        cs = sc["contact_forces"]; rb = sc["robot"]
        print("[dbg] robot bodies", rb.num_bodies, rb.body_names, flush=True)
        print("[dbg] contact bodies", len(cs.body_names), cs.body_names, flush=True)
        t = env.unwrapped.reward_manager.get_term_cfg("undesired_slippage")
        print("[dbg] slip body_ids", type(t.params["sensor_cfg"].body_ids), t.params["sensor_cfg"].body_ids, flush=True)
    obs, _ = env.reset()
    for k, v in obs.items():
        print("[smoke] obs", k, getattr(v, "shape", type(v)), flush=True)
    act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    t0 = time.time()
    n_done = 0
    for i in range(args.steps):
        obs, rew, term, trunc, _ = env.step(act)
        n_done += int((term | trunc).sum())
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[smoke] {args.steps} steps ok, {args.steps*u.num_envs/dt:.0f} env-steps/s, resets={n_done}, "
          f"rew_mean={float(rew.mean()):.3f}", flush=True)
    env.close()
