"""Shared helpers for the mapping eval scripts (no Isaac)."""

from __future__ import annotations

import os
import sys

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)

import torch  # noqa: E402

from ame2.mapping.models import load_mapping_model  # noqa: E402


def load_model(path: str, device):
    """Checkpoint dict, TorchScript file, or 'dummy'."""
    return load_mapping_model(path, device).eval()


@torch.no_grad()
def predict(model, x: torch.Tensor):
    """-> mu, sigma, gate (gate None if the model does not expose it)."""
    if hasattr(model, "forward_full"):
        try:
            mu, logvar, _, gate = model.forward_full(x)
            return mu, torch.exp(0.5 * logvar), gate
        except Exception:
            pass
    mu, logvar = model(x)
    return mu, torch.exp(0.5 * logvar), None


def model_name(path: str) -> str:
    if path == "dummy":
        return "dummy"
    p = os.path.abspath(path)
    return f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
