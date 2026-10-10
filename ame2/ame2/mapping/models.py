"""Per-frame mapping models: local grid -> (elevation estimate, log-variance).

Interface shared by every model (and by the TorchScript files the sensor loads):

    forward(x: (B, 1, L, W)) -> (mu: (B, 1, L, W), logvar: (B, 1, L, W))

``x`` is the local max-height grid of :func:`ame2.mapping.grid.points_to_local_grid`
(sensor-z-relative heights, holes = -3). ``mu`` uses the same frame.

* :class:`DummyMappingModel` -- the tuning guide's placeholder: returns the
  input as the estimate and a constant variance (0.01 by default). Combined with
  a dense top-down ray caster as the "point cloud" source this yields the
  ground-truth map, so the whole student pipeline can be built before any
  mapping model exists.
* :class:`GatedResidualUNet` -- our reading of the paper's "shallow U-Net with a
  gated residual design" (Fig. 7). Channel counts / depth are NOT given in the
  paper; the values here are our own choice. This is not the authors' model.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyMappingModel(nn.Module):
    """Identity estimate + constant variance (guide: "a constant (like 0.01) as the uncertainty")."""

    def __init__(self, variance: float = 0.01):
        super().__init__()
        self.logvar_value = float(math.log(variance))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.full_like(x, self.logvar_value)


def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
        nn.ELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ELU(),
    )


class GatedResidualUNet(nn.Module):
    """Shallow 3-level U-Net with a gated residual estimate head.

    Inputs channels: masked height, validity mask, and two normalised
    coordinate channels (the occlusion pattern depends on the distance to the
    sensor, which sits at the grid centre).

    Head outputs per cell: raw estimate ``r``, log-variance ``s``, gate logit ``g``.
    ``mu = a * x + (1 - a) * r`` with ``a = sigmoid(g) * valid`` -- observed cells
    can be passed through untouched, holes/noise are overwritten by ``r``.
    """

    def __init__(
        self,
        base_channels: int = 8,
        hole_value: float = -3.0,
        height_offset: float = 1.0,
        logvar_min: float = math.log(1e-4),
        logvar_max: float = math.log(10.0),
    ):
        super().__init__()
        c = int(base_channels)
        self.base_channels = c
        self.hole_value = float(hole_value)
        self.height_offset = float(height_offset)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.enc0 = _block(4, c)
        self.enc1 = _block(c, 2 * c, stride=2)
        self.enc2 = _block(2 * c, 4 * c, stride=2)
        self.dec1 = _block(4 * c + 2 * c, 2 * c)
        self.dec0 = _block(2 * c + c, c)
        self.head = nn.Conv2d(c, 3, 1)
        # start with an open gate and a mid-range variance
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[2] = 2.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, L, W = x.shape
        valid = (x > self.hole_value + 0.1).to(x.dtype)
        # heights are sensor-relative (ground ~ -1 m below the Mid360); shift so typical inputs are ~0
        h = (x + self.height_offset) * valid
        ys = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, L, device=x.device, dtype=x.dtype)
        cx = xs.view(1, 1, L, 1).expand(B, 1, L, W)
        cy = ys.view(1, 1, 1, W).expand(B, 1, L, W)
        inp = torch.cat([h, valid, cx, cy], dim=1)

        e0 = self.enc0(inp)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        d1 = F.interpolate(e2, size=(e1.shape[2], e1.shape[3]), mode="nearest")
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = F.interpolate(d1, size=(L, W), mode="nearest")
        d0 = self.dec0(torch.cat([d0, e0], dim=1))
        out = self.head(d0)

        raw = out[:, 0:1] - self.height_offset
        logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(out[:, 1:2])
        gate = torch.sigmoid(out[:, 2:3]) * valid
        mu = gate * x + (1.0 - gate) * raw
        return mu, logvar


# ----------------------------------------------------------------------------
# checkpoint I/O
# ----------------------------------------------------------------------------

_MODEL_REGISTRY = {"GatedResidualUNet": GatedResidualUNet}


def save_mapping_checkpoint(model: nn.Module, path: str, model_cfg: dict, extra: dict | None = None) -> None:
    """Save a plain (non-TorchScript) training checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model_class": type(model).__name__,
        "model_cfg": dict(model_cfg),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def export_mapping_jit(model: nn.Module, path: str) -> None:
    """Script + freeze a mapping model and save it for :class:`LivoxNeuralMapSensor`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    was_training = model.training
    model = model.eval().cpu()
    scripted = torch.jit.script(model)
    scripted = torch.jit.freeze(scripted)
    scripted.save(path)
    model.train(was_training)


def load_mapping_model(spec: str, device: torch.device | str = "cpu", dummy_variance: float = 0.01) -> nn.Module:
    """Load a per-frame mapping model.

    Args:
        spec: ``"dummy"`` for :class:`DummyMappingModel`, a TorchScript file
            (``torch.jit.save``), or a training checkpoint written by
            :func:`save_mapping_checkpoint` (``{"model_class", "model_cfg", "state_dict"}``).
        device: target device.
        dummy_variance: variance used by the dummy model.

    Returns:
        A module in eval mode with the shared ``forward(x) -> (mu, logvar)`` interface.
    """
    if spec == "dummy":
        return DummyMappingModel(dummy_variance).to(device).eval()
    if not os.path.isfile(spec):
        raise FileNotFoundError(
            f"Mapping model not found: {spec}. Train one with ame2/scripts/mapping/train_mapping_model.py "
            "or use the dummy mapping mode (model 'dummy' + top-down ground-truth point source)."
        )
    try:
        model = torch.jit.load(spec, map_location=device)
        return model.eval()
    except (RuntimeError, ValueError):
        pass  # not TorchScript -> plain checkpoint
    ckpt = torch.load(spec, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"{spec} is neither TorchScript nor a mapping checkpoint with a 'state_dict'.")
    cls = _MODEL_REGISTRY[ckpt.get("model_class", "GatedResidualUNet")]
    model = cls(**ckpt.get("model_cfg", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
