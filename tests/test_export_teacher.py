"""export_teacher_jit.py on the bundled modelzoo checkpoint (skipped if absent)."""

from __future__ import annotations

import os

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(REPO, "modelzoo", "g1_gaze_64000env_10h", "model_2400.pt")


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="modelzoo teacher checkpoint not present")
def test_export_modelzoo_teacher(tmp_path):
    from export_teacher_jit import export

    out = export(CKPT, str(tmp_path / "teacher.jit"), check=True)  # raises if JIT != eager
    jit = torch.jit.load(out)
    with torch.no_grad():
        a, local, glob = jit(torch.zeros(4, 82), torch.zeros(4, 26, 15, 3))
    assert a.shape == (4, 23) and local.shape == (4, 96) and glob.shape == (4, 64)
    assert torch.isfinite(a).all()
