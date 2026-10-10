"""Train the per-frame neural-mapping model: local grid -> (elevation, log-variance).

Pure PyTorch (no Isaac Sim). Data sources, mixable with ``--synthetic_ratio``:

* ``--data_dir``: shards from ``collect_mapping_data.py`` (real Mid360 pattern,
  teacher-in-the-loop poses on the locomotion terrains);
* synthetic 2.5D terrains rendered directly on the grid with simulated
  occlusion (``ame2.mapping.augment``) -- works without any recorded data.

Following the tuning guide, the first ``--clean_steps`` updates use clean
inputs (occlusion only / raw recorded scans), then the paper's augmentations
(noise, border crops, clipping, missing points, outliers) are switched on.
Loss: beta-NLL (beta = 0.5) with total-variation batch re-weighting (paper Eq. 9-10).

Outputs in ``--out_dir``: ``mapping_ckpt.pt`` (plain checkpoint, loadable by the
sensor too) and the TorchScript model, by default also copied to
``ame2/ame2/NN_models/g1LivoxMapping_perframe.pt`` (the sensor's default
``jit_model_path``).

Usage::

    # synthetic only (sanity run / bootstrap)
    python ame2/scripts/mapping/train_mapping_model.py --synthetic_ratio 1.0 --steps 20000 --device cuda
    # recorded + synthetic
    python ame2/scripts/mapping/train_mapping_model.py --data_dir data/mapping --synthetic_ratio 0.3 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch

_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

from ame2.mapping.augment import AugmentCfg, augment_inputs, make_synthetic_batch  # noqa: E402
from ame2.mapping.grid import LocalGridSpec  # noqa: E402
from ame2.mapping.losses import mapping_loss  # noqa: E402
from ame2.mapping.models import GatedResidualUNet, export_mapping_jit, save_mapping_checkpoint  # noqa: E402

DEFAULT_EXPORT = os.path.join(_EXT_DIR, "ame2", "NN_models", "g1LivoxMapping_perframe.pt")


class ShardData:
    """All recorded shards in memory (fp16), random batch sampling."""

    def __init__(self, data_dir: str, max_samples: int | None = None, val_frac: float = 0.02):
        files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if not files:
            raise FileNotFoundError(f"No shards in {data_dir}")
        xs, ys, n, spec = [], [], 0, None
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            spec = spec or d.get("spec")
            xs.append(d["x"])
            ys.append(d["y"])
            n += d["x"].shape[0]
            if max_samples and n >= max_samples:
                break
        self.x = torch.cat(xs)[:max_samples]
        self.y = torch.cat(ys)[:max_samples]
        self.spec = spec
        n = self.x.shape[0]
        n_val = max(1, int(n * val_frac))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        self.val_idx, self.train_idx = perm[:n_val], perm[n_val:]
        print(f"[INFO] loaded {n} samples from {len(files)} shard(s) ({n_val} for validation)")

    def batch(self, bs: int, device, val: bool = False):
        idx_pool = self.val_idx if val else self.train_idx
        idx = idx_pool[torch.randint(0, idx_pool.numel(), (bs,))]
        return self.x[idx].to(device).float(), self.y[idx].to(device).float()


def get_batch(args, spec, aug_cfg, data, device, clean: bool, val: bool = False, gen=None):
    """Returns (x, y, mask) each (B,1,L,W)."""
    n_syn = int(round(args.batch_size * (args.synthetic_ratio if data is not None else 1.0)))
    xs, ys, ms = [], [], []
    if n_syn > 0:
        x, y, m = make_synthetic_batch(n_syn, spec, aug_cfg, device=device, generator=gen, clean=clean)
        xs.append(x), ys.append(y), ms.append(m)
    n_rec = args.batch_size - n_syn
    if data is not None and n_rec > 0:
        x, y = data.batch(n_rec, device, val=val)
        if not clean:
            x = augment_inputs(x, spec, aug_cfg, generator=gen)
        m = y > spec.hole_value + 0.1
        xs.append(x.unsqueeze(1)), ys.append(y.unsqueeze(1)), ms.append(m.unsqueeze(1))
    return torch.cat(xs), torch.cat(ys), torch.cat(ms)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--synthetic_ratio", type=float, default=0.3, help="fraction of each batch from synthetic terrains")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--clean_steps", type=int, default=2000, help="initial steps without input corruptions")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--no_tv_reweight", action="store_true")
    p.add_argument("--base_channels", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="logs/mapping")
    p.add_argument("--export", default=DEFAULT_EXPORT, help="TorchScript export path ('' to skip)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    # grid (must match the sensor cfg)
    p.add_argument("--x_range", type=float, nargs=2, default=(-3.0, 3.0))
    p.add_argument("--y_range", type=float, nargs=2, default=(-1.5, 1.5))
    p.add_argument("--res", type=float, default=0.03)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = ShardData(args.data_dir, args.max_samples) if args.data_dir else None
    if data is not None and data.spec is not None:
        spec = LocalGridSpec(tuple(data.spec["x_range"]), tuple(data.spec["y_range"]), data.spec["res"],
                             data.spec["hole_value"], data.spec["z_max"])
    else:
        spec = LocalGridSpec(tuple(args.x_range), tuple(args.y_range), args.res)
    aug_cfg = AugmentCfg()
    print(f"[INFO] grid {spec.shape} @ {spec.res} m, device {device}")

    model_cfg = {"base_channels": args.base_channels, "hole_value": spec.hole_value}
    model = GatedResidualUNet(**model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps), eta_min=args.lr * 0.05)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] model params: {sum(p.numel() for p in model.parameters())}")

    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        clean = step <= args.clean_steps
        x, y, m = get_batch(args, spec, aug_cfg, data, device, clean)
        mu, logv = model(x)
        loss, info = mapping_loss(mu, logv, y, m, beta=args.beta, tv_reweight=not args.no_tv_reweight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        history.append(float(loss.detach()))
        if step % args.log_every == 0 or step == 1:
            print(f"[{step:6d}] loss {loss.item():.4f} beta-nll {info['beta_nll'].item():.4f} "
                  f"mae {info['mae'].item():.4f} std {info['mean_std'].item():.3f} "
                  f"{'clean ' if clean else ''}({time.time() - t0:.0f}s)")
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                g = torch.Generator(device=device).manual_seed(1234)
                xv, yv, mv = get_batch(args, spec, aug_cfg, data, device, clean=False, val=True, gen=g)
                mu_v, lv_v = model(xv)
                _, vinfo = mapping_loss(mu_v, lv_v, yv, mv, beta=args.beta, tv_reweight=False)
                # baseline: the raw grid as estimate (holes ignored)
                obs = (xv > spec.hole_value + 0.1) & mv
                raw_mae = ((xv - yv).abs() * obs).sum() / obs.sum().clamp_min(1)
            print(f"[val {step}] beta-nll {vinfo['beta_nll'].item():.4f} mae {vinfo['mae'].item():.4f} "
                  f"(raw-input mae on observed cells {raw_mae.item():.4f})")
            model.train()
            save_mapping_checkpoint(model, os.path.join(args.out_dir, "mapping_ckpt.pt"), model_cfg,
                                    extra={"step": step, "spec": spec.__dict__})

    if args.export:
        export_mapping_jit(model, args.export)
        print(f"[INFO] exported TorchScript model to {args.export}")
    export_mapping_jit(model, os.path.join(args.out_dir, "mapping_model.jit"))
    return history


if __name__ == "__main__":
    main()
