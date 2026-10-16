# AME-2 on IsaacLab 3.0 (kitless OVPhysX)

Runs the AME-2 G1 tasks on IsaacLab 3.0 with the `ovphysx` backend only (no Isaac Sim / Kit).
Environment: the IsaacLab 3.0 uv venv (`/home/zj49/workspace/IS3/.venv`, `--extra ovphysx`).
Nothing is installed into the venv: `ame2` and the vendored `rsl_rl` come from `PYTHONPATH`
(the vendored `rsl_rl` shadows the venv's `rsl-rl-lib==5.5.1`).

```bash
cd /home/zj49/nav/ame2_minimal-is3
export PYTHONPATH=$PWD/ame2:$PWD/rsl_rl
PY=/home/zj49/workspace/IS3/.venv/bin/python   # == uv run --project /home/zj49/workspace/IS3 --extra ovphysx python

# env smoke (zero actions)
$PY ame2/scripts/is3/smoke_env.py --task Ame2-G1-Gaze --num_envs 16 --steps 100 physics=ovphysx
# teacher / student training
$PY ame2/scripts/is3/train.py --task Ame2-G1-Gaze --num_envs 4096 physics=ovphysx
$PY ame2/scripts/is3/train.py --task Ame2-G1-Gaze-Student --num_envs 2048 physics=ovphysx
# layer-4 closed loop (teacher JIT; A = GT map, B = learned map, C = raw grid)
$PY ame2/scripts/is3/closed_loop_abc.py --map_mode A --num_envs 128 --out logs/is3/eval/teacher_A.json physics=ovphysx
```

`ame2/ame2/NN_models/` (teacher JIT, mapping model) is not tracked; copy it from the student worktree.

Porting notes (2.x -> 3.0): see the commit messages on `feat/isaaclab3-ovphysx`
(XYZW quaternions, ProxyArray `.torch`, warp-mask sensor API, backend factory RayCaster,
upstream backend-neutral randomization events, `(heights, origin)` height-field terrains,
`slice(None)` full resets).
