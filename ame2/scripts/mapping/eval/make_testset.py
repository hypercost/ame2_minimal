"""Layer 1a: frozen mapping test set from the Warp sampler (no Isaac / Kit).

Fixed seed, stratified by mesh family (equal frames per loaded family), augmented
inputs exactly as in training. Saved as one ``.pt`` (fp16 x / y, bool mask, family ids).

    python ame2/scripts/mapping/eval/make_testset.py --frames 20000 \\
        --locomotion_npz data/mapping/locomotion_terrain.npz --out data/mapping/testset_seed777.pt
"""

from __future__ import annotations

import argparse
import os
import time

import _common  # noqa: F401  (sys.path)
import torch

from ame2.mapping.grid import LocalGridSpec
from ame2.mapping.warp_sampler import SamplerCfg, WarpMappingSampler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=20000)
    p.add_argument("--seed", type=int, default=777, help="keep different from the training seed")
    p.add_argument("--locomotion_npz", default=None)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--rays_per_frame", type=int, default=20000)
    p.add_argument("--procedural_size", type=float, default=40.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="data/mapping/testset_seed777.pt")
    p.add_argument("--x_range", type=float, nargs=2, default=(-3.0, 3.0))
    p.add_argument("--y_range", type=float, nargs=2, default=(-1.5, 1.5))
    p.add_argument("--res", type=float, default=0.03)
    a = p.parse_args(argv)

    spec = LocalGridSpec(tuple(a.x_range), tuple(a.y_range), a.res)
    cfg = SamplerCfg(locomotion_npz=a.locomotion_npz, remesh_every=0, rays_per_frame=a.rays_per_frame,
                     procedural_size=a.procedural_size)
    s = WarpMappingSampler(spec, cfg, device=a.device, seed=a.seed)
    fams = [m.name for m in s.meshes]
    per = [a.frames // len(fams) + (1 if i < a.frames % len(fams) else 0) for i in range(len(fams))]
    xs, ys, ms, fid = [], [], [], []
    t0 = time.time()
    for i, (f, n) in enumerate(zip(fams, per)):
        done = 0
        while done < n:
            k = min(a.chunk, n - done)
            x, y, m = s.sample(k, family=f)
            xs.append(x[:, 0].half().cpu()), ys.append(y[:, 0].half().cpu()), ms.append(m[:, 0].cpu())
            fid.append(torch.full((k,), i, dtype=torch.int8))
            done += k
        print(f"[testset] {f}: {n} frames ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    torch.save({"x": torch.cat(xs), "y": torch.cat(ys), "mask": torch.cat(ms), "family": torch.cat(fid),
                "families": fams, "seed": a.seed,
                "spec": {"x_range": spec.x_range, "y_range": spec.y_range, "res": spec.res,
                         "hole_value": spec.hole_value, "z_max": spec.z_max},
                "sampler_cfg": {k: v for k, v in vars(cfg).items() if k != "aug"}}, a.out)
    print(f"[testset] wrote {a.out}: {sum(per)} frames, families {fams}")
    return a.out


if __name__ == "__main__":
    main()
