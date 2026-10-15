"""Export the locomotion training terrain mesh for the Warp mapping-data sampler.

``isaaclab.terrains`` imports ``isaaclab.sim`` (omni / pxr) at package import, so the
terrain generator cannot run without Kit. This script starts a headless app once,
builds ``AME2_TERRAINS_CFG_G1`` with Isaac Lab's ``TerrainGenerator`` (pure trimesh;
nothing is simulated) and dumps it to ``.npz`` (vertices, faces, sub-terrain origins,
walkable bounds), read by :func:`ame2.mapping.terrain_meshes.load_terrain_npz`.

Usage (Isaac python, e.g. the ``is51`` env)::

    python ame2/scripts/mapping/export_terrain_mesh.py --headless --out data/mapping/locomotion_terrain.npz
    # several difficulty seeds -> several files, pass any one to --locomotion_npz
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default="data/mapping/locomotion_terrain.npz")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--also_obj", action="store_true", help="also write a .obj next to the .npz (for viewing)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import copy  # noqa: E402

import numpy as np  # noqa: E402
from isaaclab.terrains import TerrainGenerator  # noqa: E402

from ame2.tasks.terrains import AME2_TERRAINS_CFG_G1  # noqa: E402


def main():
    cfg = copy.deepcopy(AME2_TERRAINS_CFG_G1)
    cfg.seed = args.seed
    cfg.use_cache = False
    gen = TerrainGenerator(cfg=cfg, device="cpu")
    mesh = gen.terrain_mesh
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    origins = np.asarray(gen.terrain_origins, dtype=np.float32).reshape(-1, 3)
    # walkable area = the sub-terrain grid (the border is outside)
    sx, sy = cfg.size
    half_x, half_y = 0.5 * cfg.num_rows * sx, 0.5 * cfg.num_cols * sy
    bounds = np.array([-half_x, half_x, -half_y, half_y], np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, vertices=v, faces=f, origins=origins, bounds=bounds)
    print(f"[INFO] {args.out}: {v.shape[0]} vertices, {f.shape[0]} faces, {origins.shape[0]} sub-terrains, "
          f"bounds {bounds.tolist()}")
    if args.also_obj:
        mesh.export(os.path.splitext(args.out)[0] + ".obj")


if __name__ == "__main__":
    main()
    app.close()
