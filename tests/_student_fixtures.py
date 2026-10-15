"""Shared helpers for the student / PPO-IL tests (pure torch, no isaaclab)."""

from __future__ import annotations

import torch
from tensordict import TensorDict

L, W = 26, 15          # export grid (26 x 15 @ 0.08 m)
TEACHER_PROP = 82       # G1 gaze teacher proprioception
STUDENT_PROP = 70       # any per-step student prop size works (inferred from obs)
HIST = 20
NUM_ACTIONS = 23
CRITIC_DIM = 120


def map_obs(B: int, dmap: int) -> torch.Tensor:
    xs = torch.linspace(-0.5, 1.5, L)
    ys = torch.linspace(-0.56, 0.56, W)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    chans = [gx.expand(B, L, W), gy.expand(B, L, W), 0.2 * torch.randn(B, L, W)]
    if dmap == 4:
        chans.append(0.01 + 0.1 * torch.rand(B, L, W))
    return torch.stack(chans, dim=-1)


def make_obs(B: int) -> TensorDict:
    return TensorDict(
        {
            "neural_map": map_obs(B, 4),
            "student_prop_hist": torch.randn(B, HIST, STUDENT_PROP),
            "student_cmds": torch.randn(B, 4),
            "teacher_prop": torch.randn(B, TEACHER_PROP),
            "teacher_mapping": map_obs(B, 3),
            "critic": torch.randn(B, CRITIC_DIM),
        },
        batch_size=[B],
    )


def make_teacher(seed: int = 0):
    from rsl_rl.models.AME2_models import AME2GazeModel

    torch.manual_seed(seed)
    obs = {"teacher_prop": torch.zeros(1, TEACHER_PROP), "teacher_mapping": torch.zeros(1, L, W, 3)}
    groups = {"actor": ["teacher_mapping", "teacher_prop"]}
    return AME2GazeModel(obs, groups, "actor", NUM_ACTIONS, stochastic=True, noise_std_type="log").eval()


def export_teacher(path: str, seed: int = 0) -> str:
    model = make_teacher(seed)
    torch.jit.freeze(torch.jit.script(model.as_jit().eval())).save(path)
    return path
