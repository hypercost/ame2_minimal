"""Export an AME-2 teacher checkpoint (e.g. ``modelzoo/``) to the TorchScript teacher used by PPO_IL.

Pure torch + the bundled rsl_rl -- no Isaac Sim needed. The output signature is
``teacher(teacher_prop (B, P), teacher_mapping (B, L, W, 3)) -> (actions, attn_out, global_feat)``,
exactly what :class:`rsl_rl.algorithms.ppo_il.PPO_IL` calls.

Usage (from the repo root)::

    python ame2/scripts/export_teacher_jit.py \\
        --checkpoint modelzoo/g1_gaze_64000env_10h/model_2400.pt
    # -> ame2/ame2/NN_models/g1_gaze_teacher_isaaclab.jit (the default teacher_path of the student cfgs)

The actor class and its kwargs are read from ``<run>/params/agent.yaml`` when it
exists (``--actor_class`` / CLI flags override). ``--map_shape`` must match the
teacher height-scan grid of the env (26 x 15 for the G1 tasks).
"""

from __future__ import annotations

import argparse
import os

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT = os.path.join(_REPO, "ame2", "ame2", "NN_models", "g1_gaze_teacher_isaaclab.jit")


def _read_actor_cfg(checkpoint: str) -> dict:
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "params", "agent.yaml")
    if not os.path.isfile(yaml_path):
        return {}
    try:
        import yaml
    except ImportError:
        print(f"[WARN] pyyaml not installed; ignoring {yaml_path}")
        return {}
    with open(yaml_path) as f:
        # agent.yaml files dumped by Isaac Lab may contain python tags -> unsafe loader, local file only
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    return dict(cfg.get("actor", {}) or {})


def build_teacher(checkpoint: str, actor_class: str | None = None, map_shape=(26, 15)):
    """Rebuild the eager actor from a checkpoint and return ``(model, prop_dim, dmap)``."""
    from rsl_rl.utils import resolve_callable

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["actor_state_dict"] if "actor_state_dict" in ckpt else ckpt
    actor_cfg = _read_actor_cfg(checkpoint)
    cls_name = actor_class or actor_cfg.pop("class_name", "AME2GazeModel")
    actor_cfg.pop("class_name", None)
    if ":" not in cls_name and "." not in cls_name:
        cls_name = f"rsl_rl.models.AME2_models:{cls_name}"
    cls = resolve_callable(cls_name)

    dmap = sd["fc_part.0.weight"].shape[1]
    prop_dim = sd["prop_encoder.net.0.weight"].shape[1]
    num_actions = sd["output_mlp.net.4.weight"].shape[0]
    L, W = map_shape
    obs = {"teacher_prop": torch.zeros(1, prop_dim), "teacher_mapping": torch.zeros(1, L, W, dmap)}
    groups = {"actor": ["teacher_mapping", "teacher_prop"]}
    model = cls(obs, groups, "actor", num_actions, **actor_cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch for {cls_name}: missing={missing}, unexpected={unexpected}")
    return model.eval(), prop_dim, dmap


def export(checkpoint: str, out: str, actor_class: str | None = None, map_shape=(26, 15), check: bool = True) -> str:
    model, prop_dim, dmap = build_teacher(checkpoint, actor_class, map_shape)
    jit_model = model.as_jit().cpu().eval()
    scripted = torch.jit.freeze(torch.jit.script(jit_model))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    scripted.save(out)
    print(f"[INFO] Teacher JIT written to {out} (prop_dim={prop_dim}, map={map_shape}x{dmap})")

    if check:
        L, W = map_shape
        prop = torch.randn(64, prop_dim)
        # plausible map: base-frame grid coordinates + heights
        xs = torch.linspace(-0.5, 1.5, L)
        ys = torch.linspace(-0.56, 0.56, W)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        z = 0.3 * torch.randn(64, L, W)
        tmap = torch.stack([gx.expand(64, L, W), gy.expand(64, L, W), z], dim=-1)[..., :dmap]
        loaded = torch.jit.load(out)
        with torch.no_grad():
            a_jit, l_jit, g_jit = loaded(prop, tmap)
            a_ref, l_ref, g_ref = model({"teacher_prop": prop, "teacher_mapping": tmap}, return_latent=True)
        err = max((a_jit - a_ref).abs().max().item(), (l_jit - l_ref).abs().max().item(),
                  (g_jit - g_ref).abs().max().item())
        print(f"[INFO] JIT vs eager max abs diff: {err:.2e}")
        if err > 1e-4:
            raise RuntimeError(f"Exported teacher deviates from the eager model (max abs diff {err})")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="rsl_rl checkpoint (model_*.pt) of the teacher run")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output TorchScript file (default: {DEFAULT_OUT})")
    parser.add_argument("--actor_class", default=None, help="override actor class (default: from params/agent.yaml)")
    parser.add_argument("--map_shape", type=int, nargs=2, default=(26, 15), help="teacher map grid L W")
    parser.add_argument("--no_check", action="store_true", help="skip the JIT-vs-eager consistency check")
    args = parser.parse_args()
    export(args.checkpoint, args.out, args.actor_class, tuple(args.map_shape), check=not args.no_check)


if __name__ == "__main__":
    main()
