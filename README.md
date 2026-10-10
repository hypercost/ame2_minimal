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
| Mapping model + training | not shipped | `ame2/mapping/` (gated-residual U-Net, beta-NLL + TV re-weighting, augmentations, synthetic terrains), `scripts/mapping/collect_mapping_data.py` (Isaac), `scripts/mapping/train_mapping_model.py` (pure torch) |
| Tests | rsl_rl only | `tests/` (grid projection, fusion/query, mapping model/loss/training, student models + JIT, PPO-IL updates, teacher export) |

### How to run

```bash
# 0) export the teacher used for distillation and data collection (pure torch; run with Isaac's python
#    so the TorchScript file matches the torch version that loads it)
python ame2/scripts/export_teacher_jit.py --checkpoint modelzoo/g1_gaze_64000env_10h/model_2400.pt
#    -> ame2/ame2/NN_models/g1_gaze_teacher_isaaclab.jit

# 1) student on the dummy (ground-truth) neural map -- the guide's recommended first step
./container/run.sh -g 0 train --task Ame2-G1-Gaze-Student-DummyMap --num_envs 4096
./container/run.sh -g 0 --gui play --task Ame2-G1-Gaze-Student-DummyMap-Play --checkpoint <model.pt>

# 2) collect (Mid360 grid, ground-truth grid) pairs with the teacher in the loop (inside the container)
/isaac-sim/python.sh ame2/scripts/mapping/collect_mapping_data.py --headless --num_samples 200000 --out_dir data/mapping

# 3) train the mapping model (pure torch; --data_dir optional, synthetic terrains are mixed in)
python ame2/scripts/mapping/train_mapping_model.py --data_dir data/mapping --steps 20000
#    -> logs/mapping/{mapping_ckpt.pt,mapping_model.jit} and ame2/ame2/NN_models/g1LivoxMapping_perframe.pt

# 4) student on the learned neural map
./container/run.sh -g 0 train --task Ame2-G1-Gaze-Student --num_envs 4096

# unit tests (CPU torch + tensordict + `pip install -e rsl_rl`; no Isaac needed)
python -m pytest tests
```

### Design choices / deviations (where the paper is unspecific)

- **Mapping network:** shallow 3-level gated-residual U-Net (base 8 channels); inputs = masked height,
  valid mask, 2 coordinate channels; outputs mean = gate * input + (1 - gate) * prediction and a bounded
  log-variance. Widths/depth are my own choice.
- **Loss:** beta-NLL with beta = 0.5 (Eq. 9) and total-variation batch re-weighting (Eq. 10) with a
  small TV floor (1e-3) so flat terrains are not ignored -- the floor is not from the paper.
- **Augmentations:** height noise, border crop, occlusion, clipping, missing points and outliers; ranges
  are my own. Artifacts are modelled crudely as points pulled in along the ray.
- **Synthetic data:** procedural 2.5D terrains (slopes, stairs, boxes, pits, stepping stones, rough)
  with line-of-sight occlusion, as a stand-in/supplement for the paper's mesh sampling; real Isaac
  data comes from the collection script.
- **Ground truth:** a 7 x 4 m top-down grid ray caster on the base (yaw-aligned), anchored at the Mid360
  link; isolated single-cell holes from rasterisation are filled.
- **Fusion:** unchanged from upstream. Note R_eff = max(R, 0.5 P) halves the variance per winning
  update, so with the dummy model the fused variance approaches 0.01 over a few scans, not instantly.
  Same-resolution nearest mapping local -> global can alias on even-sized global grids (some cells hit
  twice, some not at all); motion hides this.
- **Grid border:** points up to half a cell outside the grid range now land in the border cells.
- **PPO_IL:** now uses the multi-head critic (one value head per reward term), as base PPO does.

### Untested / known gaps

- Nothing has been run inside Isaac Sim: sensor integration, new env cfgs/task registration, the data
  collection script, play visualisation and full training runs are only syntax-checked.
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

