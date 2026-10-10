"""LSIO student models: shapes, latent sizes vs. the teacher, and JIT twins."""

from __future__ import annotations

import pytest
import torch

from _student_fixtures import HIST, NUM_ACTIONS, make_obs, make_teacher
from rsl_rl.models.AME2_models import AME2GazeLSIOModel, AME2LSIOModel

GROUPS = {"actor": ["neural_map", "student_prop_hist", "student_cmds"]}


@pytest.mark.parametrize("cls", [AME2GazeLSIOModel, AME2LSIOModel])
def test_student_shapes_and_latents_match_teacher(cls):
    torch.manual_seed(0)
    obs = make_obs(8)
    model = cls(obs, GROUPS, "actor", NUM_ACTIONS, stochastic=True, noise_std_type="log")
    act, local, glob = model(obs, return_latent=True, stochastic_output=True)
    assert act.shape == (8, NUM_ACTIONS)
    assert local.shape == (8, 96) and glob.shape == (8, 64)
    assert model.T == HIST

    teacher = make_teacher()
    t_obs = {"teacher_prop": obs["teacher_prop"], "teacher_mapping": obs["teacher_mapping"]}
    _, t_local, t_glob = teacher(t_obs, return_latent=True)
    assert t_local.shape == local.shape and t_glob.shape == glob.shape

    # gradients reach every map / prop branch
    (act.sum() + local.sum() + glob.sum()).backward()
    for name in ("fc_part", "cnn_part", "hist_encoder", "recent_encoder", "prop_encoder", "mha"):
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), name


@pytest.mark.parametrize("cls", [AME2GazeLSIOModel, AME2LSIOModel])
def test_student_jit_twin_matches_eager(cls, tmp_path):
    torch.manual_seed(1)
    obs = make_obs(16)
    model = cls(obs, GROUPS, "actor", NUM_ACTIONS, stochastic=True, noise_std_type="log").eval()
    with torch.no_grad():
        model(obs)  # populate model.distribution, as after training
        ref = model(obs, return_latent=True)
    path = str(tmp_path / "student.jit")
    torch.jit.freeze(torch.jit.script(model.as_jit().eval())).save(path)
    jit = torch.jit.load(path)
    with torch.no_grad():
        out = jit(obs["neural_map"], obs["student_prop_hist"], obs["student_cmds"])
    for a, b in zip(out, ref):
        assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
