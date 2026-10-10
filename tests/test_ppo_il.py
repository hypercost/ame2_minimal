"""PPO_IL (distillation -> RL) on a fake env with a scripted teacher JIT."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from _student_fixtures import NUM_ACTIONS, export_teacher, make_obs
from rsl_rl.algorithms.ppo_il import PPO_IL

NUM_ENVS, NUM_STEPS = 8, 6


def _cfg(teacher_path, rl_switch_point=3, multi_head=True):
    return {
        "num_steps_per_env": NUM_STEPS,
        "obs_groups": {"actor": ["neural_map", "student_prop_hist", "student_cmds"], "critic": ["critic"]},
        "actor": {"class_name": "rsl_rl.models.AME2_models:AME2GazeLSIOModel", "num_heads": 32,
                  "stochastic": True, "noise_std_type": "log"},
        "critic": {"class_name": "MoEModel", "num_experts": 4, "stochastic": False},
        "algorithm": {
            "class_name": "rsl_rl.algorithms.ppo_il:PPO_IL",
            "teacher_path": teacher_path,
            "rl_switch_point": rl_switch_point,
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "optimizer": "adam",
            "use_mixed_precision": False,
            "multi_head_critic": multi_head,
        },
        "multi_gpu": None,
    }


def _env(terms):
    rm = SimpleNamespace(active_terms=list(terms))
    return SimpleNamespace(num_envs=NUM_ENVS, num_actions=NUM_ACTIONS, unwrapped=SimpleNamespace(reward_manager=rm))


def _rollout(alg, obs, k):
    for _ in range(NUM_STEPS):
        alg.act(obs)
        obs = make_obs(NUM_ENVS)
        rew_c = torch.randn(NUM_ENVS, k)
        extras = {"reward_components": rew_c, "time_outs": torch.zeros(NUM_ENVS)}
        alg.process_env_step(obs, rew_c.sum(-1), torch.zeros(NUM_ENVS, dtype=torch.long), extras)
    alg.compute_returns(obs)
    return obs


@pytest.mark.parametrize("terms,optimizer,bf16", [(["a", "b", "c"], "adam", False), ([], "adam", False),
                                                  (["a", "b"], "muon", True)])
def test_ppo_il_update_before_and_after_switch(tmp_path, terms, optimizer, bf16):
    """Covers the multi-head critic (K = #reward terms), single head, and the student cfg's muon+bf16."""
    torch.manual_seed(0)
    teacher = export_teacher(str(tmp_path / "teacher.jit"))
    obs = make_obs(NUM_ENVS)
    cfg = _cfg(teacher, rl_switch_point=2)
    cfg["algorithm"].update(optimizer=optimizer, use_mixed_precision=bf16)
    alg = PPO_IL.construct_algorithm(obs, _env(terms), cfg, "cpu")
    k = max(len(terms), 1)
    assert alg.storage.rewards.shape[-1] == k

    actor0 = copy.deepcopy(alg.actor.state_dict())
    teacher_params0 = {n: t.clone() for n, t in alg.teacher.state_dict().items()}
    for it in range(2):
        obs = _rollout(alg, obs, k)
        losses = alg.update()
        for name in ("value", "imitation", "representation", "surrogate"):
            assert torch.isfinite(torch.tensor(losses[name])), (name, losses)
        assert losses["rl_coef"] == (1.0 if it == 1 else 0.0)
        if it == 0:
            assert losses["surrogate"] == 0.0
    changed = any(not torch.equal(actor0[n], p) for n, p in alg.actor.state_dict().items())
    assert changed
    for n, t in alg.teacher.state_dict().items():  # teacher frozen
        assert torch.equal(teacher_params0[n], t)


def test_distillation_reduces_imitation_loss(tmp_path):
    torch.manual_seed(0)
    teacher = export_teacher(str(tmp_path / "teacher.jit"))
    obs = make_obs(NUM_ENVS)
    alg = PPO_IL.construct_algorithm(obs, _env([]), _cfg(teacher, rl_switch_point=10_000, multi_head=False), "cpu")
    alg.imitation_coef, alg.repr_coef = 1.0, 1.0  # amplify for a short test
    fixed = make_obs(NUM_ENVS)
    hist = []
    for _ in range(12):
        for _ in range(NUM_STEPS):
            alg.act(fixed)
            alg.process_env_step(fixed, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.long), {})
        alg.compute_returns(fixed)
        hist.append(alg.update()["representation"])
    assert hist[-1] < hist[0]


def test_missing_teacher_has_actionable_error(tmp_path):
    obs = make_obs(NUM_ENVS)
    with pytest.raises(FileNotFoundError, match="export_teacher_jit"):
        PPO_IL.construct_algorithm(obs, _env([]), _cfg(str(tmp_path / "nope.jit")), "cpu")
