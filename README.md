# Minimal AME-2 extension of IsaacLab

<img src="misc/all_terrains_4x3.webp" width="100%" alt="AME-2 G1 on all training terrains">

Website: https://sites.google.com/leggedrobotics.com/ame-2  
Result Video: https://www.youtube.com/watch?v=FNA1crvtBLs  
Author: Chong Zhang (chozhang@ethz.ch or chong.zhang@ai.ethz.ch)

Please cite if you find this codebase or our paper useful:
```bibtex
@article{zhang2026agile,
  title   = {Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding},
  author  = {Zhang, Chong and Klemm, Victor and Yang, Fan and Hutter, Marco},
  journal = {IEEE Transactions on Robotics},
  year    = {2026}
}
```

The codebase is mostly vibecoded, and many IsaacLab know-hows are from Mayank Mittal. Thanks AI and Mayank! 

> **Vibe4Vibe:** the fastest way to get into this codebase is a coding agent.

> **No feature contribution:** It shall be half-archived as a reference to a finished work.


## Setup

+ Isaac Lab 2.3.2 (Isaac Sim 5.1)
+ rsl_rl: bundled in `rsl_rl/`
+ install via container (from the repo root, needs Docker + NVIDIA Container Toolkit):
  ```bash
  ./container/build.sh                                                        # build ame2-minimal:latest
  ./container/run.sh -g 0 train --task Ame2-G1-Gaze --num_envs 4096           # single-GPU training
  ./container/run.sh -g 0,1 train --task Ame2-G1-Gaze --num_envs 4096         # multi-GPU (torchrun)
  ./container/run.sh -g 0 --gui play --task Ame2-G1-Gaze-Play --checkpoint <path/to/model.pt>
  ./container/run.sh shell                                                    # interactive shell
  ./container/run.sh -d shell                                                 # background container to exec into
  ./container/run.sh exec [NAME]                                              # bash into a running container (default: newest)
  ```




## Current features:
- G1 teacher training (AME-2, optional TAGA-style active gaze, multi-headed PPO, bf16 optimization, muon optimizer)
- Neural Mapping in the loop (Mid360 lidar) -- *the official mapping model is not included; this fork adds a dummy (ground-truth) mode and a re-implemented mapping-model training pipeline, see [Fork: student & neural mapping](#fork-student--neural-mapping-re-implementation)*
- G1 student training (LSIO + AME-2, distilled + RL from a teacher JIT) -- *runnable in this fork with the dummy map or a self-trained mapping model*

- Optional new features:
    - torso, arm, and foot shaping for humanoids, not tuned. The original paper only covers legs.

## Existing checkpoints and logs:
- Check `modelzoo/`. Will add new runs in the future.

## Release Plans:
- Lidar Neural Map checkpoint
- Tron1 Env and Depth Neural Map
- Neural Map Training Code (this fork has an unofficial re-implementation, see below)

## Fork: student & neural mapping (re-implementation)

> **Unofficial.** Written from the AME-2 paper (arXiv 2601.08485, Sec. IV/V, App. B) and the
> [AME tuning guide](https://zita-ch.github.io/tech-blog/?post=2026-03-28-AME%20Tuning%20Guide.md).
> It is **not** the authors' unreleased code and is not claimed to match it. The pure-torch parts are
> unit-tested on CPU; nothing here has been run in Isaac Sim yet.

### What already existed vs. what this fork adds

| Piece | Already in upstream | Added / changed here |
|---|---|---|
| Neural-map sensor (`sensors/livox_neural_map_sensor.py`): scatter, model inference, prob. winner-take-all fusion, pose query, map DR | yes (needed the unshipped model) | `model_type="dummy"` (estimate = input, var = 0.01), `point_source="topdown_gt"` (dense top-down ray caster as noise-free point cloud), checkpoint or TorchScript loading, `em_origin_w` |
| Point cloud -> local grid | inline in the sensor | `ame2/mapping/grid.py` (`points_to_local_grid`, `scatter_reduce_` max-z), shared by sensor, data collection and tests |
| Global fusion (`neural_mapping_utils.py`) | yes, per-env python loop in `recenter` | vectorised `recenter` (gather-based, numerically identical) |
| Mid360 noise | identity placeholder | `apply_pointcloud_noise`: 15 % dropped points, 2 % artifacts (App. B); enabled for the student lidar |
| Student env / PPO_IL / LSIO models / play viz | yes | `Ame2-G1-Gaze-Student-DummyMap(-Play)` tasks; PPO_IL builds the multi-head critic like PPO; clear error if the teacher JIT is missing; JIT twin for `AME2LSIOModel` (it inherited the teacher twin) |
| Teacher JIT for distillation | not shipped | `ame2/scripts/export_teacher_jit.py` (pure torch, checks JIT == eager) |
| Mapping model + training | not shipped | `ame2/mapping/` (gated-residual U-Net, beta-NLL + TV re-weighting, augmentations), **Warp ray-tracing sampler** (`warp_sampler.py`, `terrain_meshes.py`, Kit-free), `scripts/mapping/export_terrain_mesh.py` (one-time, Kit), `scripts/mapping/train_mapping_model.py` (Warp + torch), legacy `scripts/mapping/collect_mapping_data.py` (Isaac) |
| Tests | rsl_rl only | `tests/` (grid projection, fusion/query, mapping model/loss/training, student models + JIT, PPO-IL updates, teacher export) |

### How to run

Recommended workflow (mapping data per paper Sec. V-B: Warp ray tracing, no physics):

```bash
# 0) teacher JIT for distillation (pure torch; use the torch that will load it)
python ame2/scripts/export_teacher_jit.py --checkpoint modelzoo/g1_gaze_64000env_10h/model_2400.pt

# 1) student on the dummy (ground-truth) neural map -- the guide's recommended first step
./container/run.sh -g 0 train --task Ame2-G1-Gaze-Student-DummyMap --num_envs 4096

# 2) one-time: export the locomotion training terrain mesh (needs Kit, headless, a few minutes)
python ame2/scripts/mapping/export_terrain_mesh.py --headless --out data/mapping/locomotion_terrain.npz

# 3) mapping model: online Warp sampling + training (no Kit; needs `pip install warp-lang`)
python ame2/scripts/mapping/train_mapping_model.py --benchmark          # measure frames/s -> ETA
python ame2/scripts/mapping/train_mapping_model.py --locomotion_npz data/mapping/locomotion_terrain.npz
#    default 54M frames (paper), batch 1024, bf16 on CUDA
#    -> logs/mapping/{mapping_ckpt.pt,mapping_model.jit} + ame2/ame2/NN_models/g1LivoxMapping_perframe.pt

# 4) student on the learned neural map
./container/run.sh -g 0 train --task Ame2-G1-Gaze-Student --num_envs 4096

# optional / legacy: record in-sim Mid360 frames with the teacher, then fine-tune on them
/isaac-sim/python.sh ame2/scripts/mapping/collect_mapping_data.py --headless --num_samples 200000 --out_dir data/mapping/shards
python ame2/scripts/mapping/train_mapping_model.py --source shards --data_dir data/mapping/shards --frames 5e6

# unit tests (CPU torch + tensordict + warp-lang + `pip install -e rsl_rl`; no Isaac needed)
python -m pytest tests
```

On a local conda Isaac env (e.g. `is51`, torch 2.7 + cu128), long runs can be detached:

```bash
conda activate is51
python -c "import warp" || pip install warp-lang   # the Kit-free script needs the warp python package
mkdir -p logs/mapping
setsid nohup python ame2/scripts/mapping/train_mapping_model.py \
    --locomotion_npz data/mapping/locomotion_terrain.npz --out_dir logs/mapping \
    > logs/mapping/train.log 2>&1 < /dev/null &
tail -f logs/mapping/train.log
```

`ame2` no longer auto-imports the Isaac tasks/sensors unless a Kit app is running, so the
mapping tools work in an env where `isaaclab` is installed but Kit is not started.

**Run time.** The paper reports < 1 h per robot for 54M frames on its 51x31 grid (sampling on a
single RTX 4090 at "hundreds of thousands of frames per second"). The G1 grid here is 201x101
(13x more cells, 20k rays per frame), so expect it to be slower; `--benchmark` prints the measured
frames/s and the resulting ETA for your GPU. Measured here on CPU only (8 cores, no GPU):
~41 frames/s sampling + ~119 frames/s training at batch 64 -- i.e. CPU is only for tests.

### Mapping-model validation (4 layers)

Scripts in `ame2/scripts/mapping/eval/`, metrics in `ame2/mapping/eval_metrics.py` (pure torch, unit-tested).
Layers 1-2 run without Kit; layers 3-4 need Isaac Sim + GPU and **have not been run yet**.
Long runs: prefix with `setsid nohup ... > log 2>&1 < /dev/null &` (conda env `is51`); all four fit easily in
~64 GB of GPU memory at the settings below.

```bash
conda activate is51
M=ame2/ame2/NN_models/g1LivoxMapping_perframe.pt          # or logs/mapping/mapping_ckpt.pt
# Layer 1: frozen test set (once), then evaluate / compare checkpoints
python ame2/scripts/mapping/eval/make_testset.py --frames 20000 --seed 777 \
    --locomotion_npz data/mapping/locomotion_terrain.npz --out data/mapping/testset_seed777.pt
python ame2/scripts/mapping/eval/eval_offline.py --testset data/mapping/testset_seed777.pt \
    --checkpoints $M dummy --out_dir logs/mapping_eval/offline
# Layer 2: unit scenes (box shadow, stair edge, stepping stones, floating box)
python ame2/scripts/mapping/eval/unit_scenes.py --checkpoint $M --out_dir logs/mapping_eval/scenes
# Layer 3: in-loop fused map vs GT (teacher drives on GT; Mid360 + learned model build the map)
setsid nohup python ame2/scripts/mapping/eval/isaac_map_vs_gt.py --headless --num_envs 256 --steps 3000 \
    --mapping_model $M --out_dir logs/mapping_eval/inloop_B > logs/mapping_eval/inloop_B.log 2>&1 < /dev/null &
# Layer 4: closed loop A (GT) / B (learned) / C (raw grid + const var), same seed, one process per mode
for MODE in A B C; do
  python ame2/scripts/mapping/eval/closed_loop_abc.py --headless --map_mode $MODE --num_envs 512 \
      --episodes_per_env 2 --seed 0 --mapping_model $M --out logs/mapping_eval/abc/teacher_$MODE.json
done
python ame2/scripts/mapping/eval/compare_abc.py logs/mapping_eval/abc/teacher_*.json
# final: the same with a student policy JIT (play.py --export) instead of the teacher
#   ... closed_loop_abc.py --policy student --student_jit <exported_student.jit> --out .../student_$MODE.json
```

Pass criteria (my proposal; thresholds are not from the paper):

| Layer | Pass if |
|---|---|
| 1 offline | observed-cell MAE <= raw-input MAE; hole MAE < min(mean-fill, nearest-fill) overall **and** in each distance bucket (1-3 / 4-8 / >8 cells); within-1σ in [0.55, 0.85] and within-2σ in [0.85, 0.99]; AUSE < 0.5 x the AUSE of a random uncertainty ranking. `summary.md` prints PASS/FAIL per criterion. |
| 2 scenes | box: mean σ in the shadow >= 2 x σ on visible ground and >= 0.1 m; stair: predicted max step across the edge within ±30 % of the GT 0.17 m; stones: >= 90 % of gap cells predicted > 0.15 m below the stone top or σ > 0.15 m (σ only counts if σ on the stone tops is <= 0.15 m; the dummy model passes this one trivially because holes stay at -3); floating box: report only. All three must pass. |
| 3 in-loop | observed-cell MAE < 0.05 m; MAE while turning (|yaw rate| > 0.5 rad/s) <= 1.5 x straight; no terrain with jump rate (|ΔMAE| > 5 cm per step) far above the others. |
| 4 closed loop | B success within 5 points of A overall and on stones / gap / beam / pallets, B fall rate <= A + 5 points, and B clearly better than C on those terrains. |

**Teacher vs. student map observations (checked at runtime by `check_map_compat`).** Both are 26 x 15 grids
over base-frame x in [-0.5, 1.5], y in [-0.56, 0.56] at 0.08 m, x-major, z = height − base z + 0.74. Differences:
student z is clipped to [-2, 2] (teacher [-1.5, 1.5]); the student has a 4th uncertainty channel; unseen cells
are the -2 clip in the student map, while teacher voids are filled with the lowest valid height minus 0.65-1.5 m.
For the teacher in modes B/C, `neural_to_teacher_map` drops the uncertainty and clips z to [-1.5, 1.5] (unseen
cells become -1.5, a pit, similar to teacher voids). The teacher therefore ignores σ entirely, which penalises
B and C relative to a student that was trained to use σ. For a student in mode A, the GT map gets a confident
variance (0.01). Mode C = dummy model on the Mid360 points: hole pixels are never fused (same rule as for the
learned model), observed pixels win with variance 0.01.

### Design choices / deviations (where the paper is unspecific)

- **Mapping network:** shallow 3-level gated-residual U-Net (base 8 channels); inputs = masked height,
  valid mask, 2 coordinate channels; outputs mean = gate * input + (1 - gate) * prediction and a bounded
  log-variance. Widths/depth are my own choice.
- **Loss:** beta-NLL with beta = 0.5 (Eq. 9) and total-variation batch re-weighting (Eq. 10) with a
  small TV floor (1e-3) so flat terrains are not ignored -- the floor is not from the paper.
- **Augmentations:** height noise, border crop, occlusion, clipping, missing points and outliers; ranges
  are my own. Artifacts are modelled crudely as points pulled in along the ray.
- **Training data (paper Sec. V-B1/2):** `ame2/mapping/warp_sampler.py` samples random G1 poses
  (Mid360 0.9-1.45 m above the ground below it, yaw uniform, roll/pitch <= 10 deg) above four
  mesh families (`terrain_meshes.py`: exported locomotion terrains, stacked boxes, random height
  fields, floating boxes; mixing weights 0.4/0.2/0.2/0.2, procedural meshes rebuilt every 2000
  batches). Label = top-down ray per cell starting 2.5 m above the sensor (like the in-sim GT
  scanner), clamped to [-3, 2]. Input = rays from a jittered sensor pose (+-0.1 m xy, +-0.05 m z),
  50 % the real Mid360 pattern (random 20k-ray window, sim mount), 50 % a random FOV (90-360 deg
  azimuth sector, random elevation band); visible hits -> max-z grid (geometric occlusion), then
  noise / border crop / clipping / missing points / outliers. All ranges are my own; the paper gives
  none. The grid stays the sensor's G1 grid (201x101 @ 0.03 m, centred on the Mid360, base-yaw
  axes), not the paper's ANYmal 51x31 @ 4 cm. The old 2.5D synthetic terrains remain as
  `--source synthetic`.
- **Ground truth:** a 7 x 4 m top-down grid ray caster on the base (yaw-aligned), anchored at the Mid360
  link; isolated single-cell holes from rasterisation are filled.
- **Fusion:** unchanged from upstream. Note R_eff = max(R, 0.5 P) halves the variance per winning
  update, so with the dummy model the fused variance approaches 0.01 over a few scans, not instantly.
  Same-resolution nearest mapping local -> global can alias on even-sized global grids (some cells hit
  twice, some not at all); motion hides this.
- **Grid border:** points up to half a cell outside the grid range now land in the border cells.
- **PPO_IL:** now uses the multi-head critic (one value head per reward term), as base PPO does.

### Untested / known gaps

- Nothing has been run inside Isaac Sim: sensor integration, new env cfgs/task registration, the terrain
  exporter, the legacy collection script, play visualisation and full training runs are only syntax-checked.
- The Warp sampler was only run on CPU (warp CPU backend); GPU throughput is unmeasured.
- Validation layers 3 and 4 (Isaac) are unrun; only their pure logic is unit-tested.
- The mapping model was only trained for short CPU sanity runs on synthetic data; no checkpoint is shipped.
- The TorchScript teacher is not committed: export it with the torch version that will load it.
- Tron1 and the depth neural map are not implemented.

## Notes:
- Some trivial details might not be perfectly aligned between IsaacLab and legged gym. Some might be improved.
- The results for humanoids do not represent the limits of our method (contributions mostly on generalization and neural mapping) -- there is no targetted performance tuning or style optimization, especially for G1.
- MDP setup shall easily transfer to other robots.

## License

© 2026 ETH Zurich  
Created by: Chong Zhang, Victor Klemm, Fan Yang, and Marco Hutter  
Licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)).

Third-party code keeps its original license (see [licenses/dependencies](licenses/dependencies)):
- `rsl_rl/` is a modified copy of [rsl_rl](https://github.com/leggedrobotics/rsl_rl), BSD-3-Clause.
- Files in `ame2/` marked "The Isaac Lab Project Developers" are derived from [Isaac Lab](https://github.com/isaac-sim/IsaacLab), BSD-3-Clause. Isaac Lab itself is not included.
- `ame2/ame2/sensors/mid360_raydirs.npy` (Livox Mid-360 scan pattern) comes from [fratopa/Mid360_simulation_plugin](https://github.com/fratopa/Mid360_simulation_plugin).

