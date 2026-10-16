"""Isaac-side helpers shared by layers 3 / 4. Import only AFTER the simulation was launched."""

from __future__ import annotations

import os

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import ame2.tasks  # noqa: F401  (registers the gym ids)
from ame2.mapping.eval_metrics import apply_map_mode, terrain_column_names

EVAL_EVENTS_OFF = ("physics_material", "add_base_mass", "base_com", "randomize_actuator_gains", "push_robot")


def make_eval_env(task: str, num_envs: int, device: str, seed: int, map_mode: str, mapping_model: str | None,
                  terrain: str | None = None, difficulty: float | None = None, video: bool = False,
                  video_dir: str | None = None, video_length: int = 400, env_cfg=None):
    """Student-style env with deterministic eval settings and the requested map mode.

    ``env_cfg``: an already resolved config (IsaacLab 3.0 scripts resolve presets via Hydra first).
    """
    if env_cfg is None:
        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    else:
        env_cfg.scene.num_envs = num_envs
        if device is not None:
            env_cfg.sim.device = device
    env_cfg.seed = seed
    if not hasattr(env_cfg.scene, "neural_map") or env_cfg.scene.neural_map is None:
        raise ValueError(f"{task} has no neural_map sensor; use a student task (e.g. Ame2-G1-Gaze-Student)")
    if getattr(env_cfg.scene, "lidar_cam", None) is None:
        raise ValueError(f"{task} has no Mid360 lidar (dummy-map tasks drop it); use Ame2-G1-Gaze-Student")
    if mapping_model and not os.path.isabs(mapping_model):
        mapping_model = os.path.abspath(mapping_model)
    mode_info = apply_map_mode(env_cfg.scene.neural_map, map_mode, mapping_model)
    # ablations: AME2_NM_OVERRIDES="filter_estimation_std=0.15;abs_accept_var=0.04"
    import ast
    for kv in filter(None, os.environ.get("AME2_NM_OVERRIDES", "").split(";")):
        k, v = kv.split("=", 1)
        setattr(env_cfg.scene.neural_map, k.strip(), ast.literal_eval(v.strip()))
        mode_info[f"nm.{k.strip()}"] = v.strip()
    # deterministic eval: no DR events, no curricula (terrain levels stay at their random init)
    for name in EVAL_EVENTS_OFF:
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    for name in list(vars(env_cfg.curriculum)):
        if not name.startswith("_"):
            setattr(env_cfg.curriculum, name, None)
    if hasattr(env_cfg.actions, "joint_pos") and hasattr(env_cfg.actions.joint_pos, "delay_range"):
        env_cfg.actions.joint_pos.delay_range = (0, 0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.observations.teacher_mapping.enable_corruption = False
    env_cfg.observations.teacher_prop.enable_corruption = False
    if hasattr(env_cfg.observations, "neural_map"):
        env_cfg.observations.neural_map.enable_corruption = False
    tg = env_cfg.scene.terrain.terrain_generator
    if tg is not None:
        tg.seed = seed
        if terrain is not None:
            if terrain not in tg.sub_terrains:
                raise ValueError(f"unknown terrain {terrain!r}: {sorted(tg.sub_terrains)}")
            sel = tg.sub_terrains[terrain]
            sel.proportion = 1.0
            tg.sub_terrains = {terrain: sel}
        if difficulty is not None:
            tg.difficulty_range = (difficulty, difficulty)
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if video else None)
    if video:
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, step_trigger=lambda s: s == 0,
                                       video_length=video_length, disable_logger=True)
    return env, env_cfg, mode_info


def env_terrain_names(uenv) -> list[str]:
    terr = uenv.scene.terrain
    tg = terr.cfg.terrain_generator
    if tg is None or not hasattr(terr, "terrain_types"):
        return ["plane"] * uenv.num_envs
    cols = terrain_column_names(tg.sub_terrains, tg.num_cols)
    return [cols[int(t)] if int(t) < len(cols) else f"Type_{int(t)}" for t in terr.terrain_types.tolist()]


def load_policy(kind: str, path: str, device):
    """-> callable(obs, map_tensor) -> actions. teacher JIT: (prop, map); student JIT: (map, prop_hist, cmds)."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{kind} JIT not found: {path}")
    net = torch.jit.load(path, map_location=device).eval()
    if kind == "teacher":
        return lambda obs, m: net(obs["teacher_prop"], m)[0]
    if kind == "student":
        return lambda obs, m: net(m, obs["student_prop_hist"], obs["student_cmds"])[0]
    raise ValueError(kind)
