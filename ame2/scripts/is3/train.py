"""Train AME-2 with the vendored RSL-RL on IsaacLab 3.0 (kitless OVPhysX by default).

Uses the repo's own ``rsl_rl`` (put it first on PYTHONPATH) instead of the venv's rsl-rl-lib, like
``scripts/rsl_rl/train.py`` on the 2.x stack, but launches through ``launch_simulation`` and resolves
Hydra presets (``physics=ovphysx``) via ``resolve_task_config``.

  cd <repo> && PYTHONPATH=$PWD/ame2:$PWD/rsl_rl /home/zj49/workspace/IS3/.venv/bin/python \
      ame2/scripts/is3/train.py --task Ame2-G1-Gaze --num_envs 4096 physics=ovphysx
"""

import argparse
import os
import sys
import time
from datetime import datetime

from isaaclab.app import add_launcher_args, launch_simulation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # noqa: E402  (ame2/scripts/rsl_rl/cli_args.py)

parser = argparse.ArgumentParser(description="Train an AME-2 agent with RSL-RL (IsaacLab 3.0).")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)

import ame2.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli  # noqa: E402

args_cli, rest = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + rest
env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym
    import torch
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)
    env_cfg.log_dir = log_dir

    resume_path = None
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    start = time.time()
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}", flush=True)
        runner.load(resume_path)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start, 2)} seconds", flush=True)
    env.close()
