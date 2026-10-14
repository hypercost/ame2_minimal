"""Pure torch/numpy evaluation logic for the mapping model (validation layers 1-4).

No Isaac imports: the Isaac scripts (layers 3/4) only collect tensors and call
into this module, so everything here is unit-tested on CPU.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HOLE_BUCKETS = ((1, 3), (4, 8), (9, 10**9))      # distance (cells) from the nearest observed cell
BUCKET_NAMES = ("1-3", "4-8", ">8")


# ----------------------------------------------------------------------------
# Layer 1: baselines, hole buckets, calibration, sparsification
# ----------------------------------------------------------------------------

def observed_mask(x: torch.Tensor, hole_value: float = -3.0) -> torch.Tensor:
    return x > hole_value + 0.1


def mean_fill(x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Holes <- mean of the observed cells of the same frame. x, obs: (B,1,L,W)."""
    n = obs.sum((2, 3), keepdim=True).clamp_min(1)
    m = (x * obs).sum((2, 3), keepdim=True) / n
    return torch.where(obs, x, m.expand_as(x))


def nearest_fill(x: torch.Tensor, obs: torch.Tensor, max_iters: int = 400) -> torch.Tensor:
    """Onion-peel inpainting: each pass fills the hole cells touching filled cells with the
    mean of their filled 3x3 neighbours (a nearest-neighbour fill with local averaging)."""
    v = torch.where(obs, x, torch.zeros_like(x))
    f = obs.to(x.dtype)
    k = torch.ones(1, 1, 3, 3, dtype=x.dtype, device=x.device)
    for _ in range(max_iters):
        if bool(f.min() > 0):
            break
        s = F.conv2d(v * f, k, padding=1)
        c = F.conv2d(f, k, padding=1)
        new = (f == 0) & (c > 0)
        v = torch.where(new, s / c.clamp_min(1e-6), v)
        f = torch.where(new, torch.ones_like(f), f)
    # frames without any observation: fall back to 0 (sensor height)
    return torch.where(f > 0, v, torch.zeros_like(v))


def distance_to_observed(obs: torch.Tensor, max_dist: int = 64) -> torch.Tensor:
    """Chebyshev distance (cells) of every cell to the nearest observed cell (0 on observed).
    Cells farther than ``max_dist`` get ``max_dist + 1``."""
    d = torch.full(obs.shape, max_dist + 1, dtype=torch.int32, device=obs.device)
    reached = obs.clone()
    d[reached] = 0
    cur = obs.to(torch.float32)
    for i in range(1, max_dist + 1):
        cur = F.max_pool2d(cur, 3, stride=1, padding=1)
        new = (cur > 0) & ~reached
        if not bool(new.any()):
            break
        d[new] = i
        reached |= new
    return d


def sparsification(err: torch.Tensor, sigma: torch.Tensor, n_points: int = 20, max_cells: int = 2_000_000,
                   seed: int = 0) -> dict:
    """Sparsification curve (remove the most uncertain fraction, MAE of the rest) vs. the oracle
    (remove the largest errors). AUSE = area between the two (fractions 0 .. 1 - 1/n_points)."""
    err, sigma = err.reshape(-1).float(), sigma.reshape(-1).float()
    if err.numel() > max_cells:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(err.numel(), generator=g)[:max_cells].to(err.device)
        err, sigma = err[idx], sigma[idx]
    n = err.numel()
    fracs = np.linspace(0.0, 1.0 - 1.0 / n_points, n_points)
    by_sigma = err[torch.argsort(sigma, descending=True)]
    by_err = torch.sort(err, descending=True).values
    cs_s = torch.cumsum(by_sigma.flip(0), 0).flip(0)    # suffix sums: mean of the kept tail
    cs_o = torch.cumsum(by_err.flip(0), 0).flip(0)
    curve, oracle = [], []
    for f in fracs:
        k = min(int(round(f * n)), n - 1)
        curve.append(float(cs_s[k] / (n - k)))
        oracle.append(float(cs_o[k] / (n - k)))
    curve, oracle = np.array(curve), np.array(oracle)
    trap = getattr(np, "trapezoid", None) or np.trapz
    ause = float(trap(curve - oracle, fracs))
    return {"fractions": fracs.tolist(), "model": curve.tolist(), "oracle": oracle.tolist(), "ause": ause}


def calibration(err: torch.Tensor, sigma: torch.Tensor, ks=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)) -> dict:
    """Observed fraction of |err| <= k * sigma vs. the Gaussian expectation erf(k / sqrt 2)."""
    err, sigma = err.reshape(-1).float(), sigma.reshape(-1).float()
    obs = [float((err <= k * sigma).float().mean()) if err.numel() else float("nan") for k in ks]
    exp = [math.erf(k / math.sqrt(2.0)) for k in ks]
    return {"k": list(ks), "observed": obs, "expected": exp,
            "within_1sigma": obs[list(ks).index(1.0)], "within_2sigma": obs[list(ks).index(2.0)]}


class OfflineAccumulator:
    """Streams batches of (input, label, mask, mu, sigma[, gate]) and builds the layer-1 summary."""

    def __init__(self, hole_value: float = -3.0, clean_tol: float = 0.05, max_cells: int = 2_000_000):
        self.hole = hole_value
        self.clean_tol = clean_tol
        self.max_cells = max_cells
        self.s = defaultdict(float)
        self._err, self._sig = [], []
        self._n_kept = 0

    def _add(self, key, values: torch.Tensor):
        self.s[key + "/sum"] += float(values.sum())
        self.s[key + "/n"] += float(values.numel())

    @torch.no_grad()
    def update(self, x, y, m, mu, sigma, gate=None):
        x, y, mu, sigma = x.float(), y.float(), mu.float(), sigma.float()
        m = m.bool()
        obs = observed_mask(x, self.hole)
        holes = ~obs & m
        o = obs & m
        e = (mu - y).abs()
        self._add("model/observed", e[o])
        self._add("raw/observed", (x - y).abs()[o])
        self._add("model/hole", e[holes])
        mf = mean_fill(x, obs)
        nf = nearest_fill(x, obs)
        self._add("mean_fill/hole", (mf - y).abs()[holes])
        self._add("nearest_fill/hole", (nf - y).abs()[holes])
        self._add("model/all", e[m])
        d = distance_to_observed(obs)
        for (lo, hi), name in zip(HOLE_BUCKETS, BUCKET_NAMES):
            b = holes & (d >= lo) & (d <= hi)
            self._add(f"model/hole_{name}", e[b])
            self._add(f"mean_fill/hole_{name}", (mf - y).abs()[b])
            self._add(f"nearest_fill/hole_{name}", (nf - y).abs()[b])
        self._add("cells/observed", o.float())
        self._add("cells/hole", holes.float())
        for k in (1.0, 2.0):
            self._add(f"calib/within_{k:g}sigma", (e[m] <= k * sigma[m]).float())
        if gate is not None:
            g = gate.float()
            clean = o & ((x - y).abs() <= self.clean_tol)
            corrupt = o & ((x - y).abs() > self.clean_tol)
            self._add("gate/observed_clean", g[clean])
            self._add("gate/observed_corrupted", g[corrupt])
            self._add("gate/hole", g[holes])
        # reservoir for calibration / sparsification
        ee, ss = e[m].reshape(-1), sigma[m].reshape(-1)
        room = self.max_cells - self._n_kept
        if room > 0:
            if ee.numel() > room:
                idx = torch.randperm(ee.numel(), device=ee.device)[:room]
                ee, ss = ee[idx], ss[idx]
            self._err.append(ee.cpu())
            self._sig.append(ss.cpu())
            self._n_kept += ee.numel()

    def mean(self, key):
        n = self.s.get(key + "/n", 0.0)
        return self.s[key + "/sum"] / n if n > 0 else float("nan")

    def summary(self) -> dict:
        out = {
            "observed_mae": {"model": self.mean("model/observed"), "raw_input": self.mean("raw/observed")},
            "hole_mae": {"model": self.mean("model/hole"), "mean_fill": self.mean("mean_fill/hole"),
                         "nearest_fill": self.mean("nearest_fill/hole")},
            "hole_mae_by_distance": {
                name: {"model": self.mean(f"model/hole_{name}"), "mean_fill": self.mean(f"mean_fill/hole_{name}"),
                       "nearest_fill": self.mean(f"nearest_fill/hole_{name}"),
                       "cells": self.s.get(f"model/hole_{name}/n", 0.0)}
                for name in BUCKET_NAMES},
            "all_mae": self.mean("model/all"),
            "observed_fraction": self.mean("cells/observed") / max(
                self.mean("cells/observed") + self.mean("cells/hole"), 1e-12)
            if self.s.get("cells/observed/n") else float("nan"),
            "within_1sigma": self.mean("calib/within_1sigma"),
            "within_2sigma": self.mean("calib/within_2sigma"),
        }
        if "gate/observed_clean/n" in self.s:
            out["gate"] = {k: self.mean(f"gate/{k}") for k in ("observed_clean", "observed_corrupted", "hole")}
        if self._err:
            err, sig = torch.cat(self._err), torch.cat(self._sig)
            out["calibration"] = calibration(err, sig)
            out["sparsification"] = sparsification(err, sig)
            out["ause"] = out["sparsification"]["ause"]
            # reference: a random uncertainty ranking (what a constant sigma amounts to)
            g = torch.Generator(device="cpu").manual_seed(0)
            out["ause_random"] = sparsification(err, torch.rand(err.shape, generator=g))["ause"]
        return out


# ----------------------------------------------------------------------------
# Layer 2: unit-scene checks (operate on 2D numpy/torch grids, x = forward axis = dim 0)
# ----------------------------------------------------------------------------

def check_shadow_sigma(sigma, shadow_mask, visible_mask, min_ratio: float = 2.0, min_sigma: float = 0.1) -> dict:
    s_sh = float(sigma[shadow_mask].mean())
    s_vis = float(sigma[visible_mask].mean())
    ok = (s_sh >= min_ratio * s_vis) and (s_sh >= min_sigma)
    return {"pass": bool(ok), "sigma_shadow": s_sh, "sigma_visible": s_vis, "min_ratio": min_ratio,
            "min_sigma": min_sigma}


def max_step_along_x(h, band_mask) -> float:
    """Largest |h[i+1] - h[i]| along x inside ``band_mask`` (both cells in the band)."""
    h = torch.as_tensor(h, dtype=torch.float32)
    b = torch.as_tensor(band_mask)
    d = (h[1:] - h[:-1]).abs()
    both = b[1:] & b[:-1]
    return float(d[both].max()) if bool(both.any()) else 0.0


def check_edge_sharpness(pred, gt, band_mask, tol: float = 0.3) -> dict:
    """Pass if the predicted max step across the edge keeps >= (1 - tol) of the GT step and does
    not overshoot it by more than tol."""
    g = max_step_along_x(gt, band_mask)
    p = max_step_along_x(pred, band_mask)
    ratio = p / g if g > 0 else float("nan")
    return {"pass": bool(g > 0 and (1 - tol) <= ratio <= (1 + tol)), "gt_max_step": g, "pred_max_step": p,
            "ratio": ratio, "tol": tol}


def check_gaps_not_filled(pred, sigma, gap_mask, stone_top: float, margin: float = 0.15,
                          sigma_thr: float = 0.15, min_frac: float = 0.9, stone_mask=None) -> dict:
    """Gap cells must stay clearly below the stone top OR be flagged uncertain.

    If ``stone_mask`` is given and sigma on the stones is itself above ``sigma_thr`` (a model that is
    uncertain everywhere), high sigma does not count -- only the height condition does.
    """
    p = torch.as_tensor(pred, dtype=torch.float32)[torch.as_tensor(gap_mask)]
    s = torch.as_tensor(sigma, dtype=torch.float32)[torch.as_tensor(gap_mask)]
    stone_sigma = float("nan")
    sigma_counts = True
    if stone_mask is not None and bool(torch.as_tensor(stone_mask).any()):
        stone_sigma = float(torch.as_tensor(sigma, dtype=torch.float32)[torch.as_tensor(stone_mask)].mean())
        sigma_counts = stone_sigma <= sigma_thr
    low = p < stone_top - margin
    ok_cells = (low | (s > sigma_thr)) if sigma_counts else low
    frac = float(ok_cells.float().mean()) if p.numel() else float("nan")
    return {"pass": bool(p.numel() > 0 and frac >= min_frac), "fraction_ok": frac, "min_frac": min_frac,
            "fraction_below_top": float(low.float().mean()) if p.numel() else float("nan"),
            "stone_top": stone_top, "margin": margin, "sigma_thr": sigma_thr, "sigma_on_stones": stone_sigma,
            "sigma_counts": sigma_counts, "gap_pred_mean": float(p.mean()) if p.numel() else float("nan")}


# ----------------------------------------------------------------------------
# Layers 3/4: observation compatibility, map modes, terrain names, aggregators
# ----------------------------------------------------------------------------

STUDENT_ELEVATION_OFFSET = 0.74     # ame2_neural_map_obs_from_sensor(elevation_offset) in the student cfgs
TEACHER_OFFSET = 0.74               # ame2_teacher_mapping(offset)
TEACHER_CLIP = (-1.5, 1.5)          # ame2_teacher_mapping(clip_z)
STUDENT_CLIP = (-2.0, 2.0)
UNC_CONFIDENT = 0.5 * math.log(0.01) + 0.5   # obs value of variance 0.01


def check_map_compat(neural_map: torch.Tensor, teacher_map: torch.Tensor, atol: float = 1e-4) -> dict:
    """Student map obs (B,L,W,4: x,y,z,unc) vs. teacher map obs (B,L,W,3: x,y,z)."""
    issues = []
    if neural_map.dim() != 4 or neural_map.shape[-1] != 4:
        issues.append(f"neural_map shape {tuple(neural_map.shape)} is not (B,L,W,4)")
    if teacher_map.dim() != 4 or teacher_map.shape[-1] != 3:
        issues.append(f"teacher_mapping shape {tuple(teacher_map.shape)} is not (B,L,W,3)")
    if not issues and neural_map.shape[1:3] != teacher_map.shape[1:3]:
        issues.append(f"grid mismatch {tuple(neural_map.shape[1:3])} vs {tuple(teacher_map.shape[1:3])}")
    if not issues:
        dxy = (neural_map[..., :2] - teacher_map[..., :2]).abs().max().item()
        if dxy > atol:
            issues.append(f"x/y channels differ by up to {dxy:.4g} (teacher x/y noise or different grid order)")
    return {"compatible": not issues, "issues": issues,
            "shape_student": tuple(neural_map.shape), "shape_teacher": tuple(teacher_map.shape)}


def neural_to_teacher_map(neural_map: torch.Tensor, clip=TEACHER_CLIP) -> torch.Tensor:
    """Student map obs -> teacher map obs format. Both z channels are base-relative + 0.74, so only
    the clip differs ([-2, 2] -> [-1.5, 1.5]); the uncertainty channel is dropped (the teacher
    has none). Unseen cells (student z = -2) become -1.5, i.e. a pit, as teacher voids are."""
    z = (neural_map[..., 2] - STUDENT_ELEVATION_OFFSET + TEACHER_OFFSET).clamp(*clip)
    return torch.stack([neural_map[..., 0], neural_map[..., 1], z], -1)


def teacher_to_neural_map(teacher_map: torch.Tensor, unc_obs: float = UNC_CONFIDENT) -> torch.Tensor:
    """GT teacher map -> student map obs (for mode A with a student policy): confident variance 0.01."""
    z = (teacher_map[..., 2] - TEACHER_OFFSET + STUDENT_ELEVATION_OFFSET).clamp(*STUDENT_CLIP)
    u = torch.full_like(z, unc_obs)
    return torch.stack([teacher_map[..., 0], teacher_map[..., 1], z, u], -1)


MAP_MODES = {
    "A": {"label": "GT map", "model_type": "dummy", "point_source": "lidar", "feed": "teacher_mapping"},
    "B": {"label": "learned mapping", "model_type": "learned", "point_source": "lidar", "feed": "neural_map"},
    "C": {"label": "raw grid + const var", "model_type": "dummy", "point_source": "lidar", "feed": "neural_map"},
}


def apply_map_mode(neural_map_cfg, mode: str, mapping_model: str | None = None, dummy_variance: float = 0.01) -> dict:
    """Configure a LivoxNeuralMapSensorCfg-like object for map mode A/B/C and turn off map DR.

    A: the policy is fed the GT ``teacher_mapping`` obs; the neural map sensor still runs (dummy,
       cheap) so the scene is identical across modes.
    B: learned mapping model on the Mid360 points.
    C: dummy model on the Mid360 points = raw max-z grid + constant variance (holes never fused).
    """
    if mode not in MAP_MODES:
        raise ValueError(f"map mode must be one of {sorted(MAP_MODES)}, got {mode!r}")
    spec = MAP_MODES[mode]
    neural_map_cfg.model_type = spec["model_type"]
    neural_map_cfg.point_source = spec["point_source"]
    neural_map_cfg.dummy_variance = dummy_variance
    if mode == "B":
        if not mapping_model:
            raise ValueError("map mode B needs --mapping_model")
        neural_map_cfg.jit_model_path = mapping_model
    # evaluation: no map domain randomisation
    for name, val in (("query_drift_range", 0.0), ("clean_source_ratio", 0.0), ("random_pixel_ratio", 0.0)):
        if hasattr(neural_map_cfg, name):
            setattr(neural_map_cfg, name, val)
    return dict(spec)


def policy_map_input(mode: str, obs: dict, policy: str) -> torch.Tensor:
    """Map tensor fed to the policy for a mode. policy: 'teacher' -> (B,L,W,3); 'student' -> (B,L,W,4)."""
    feed = MAP_MODES[mode]["feed"]
    if policy == "teacher":
        return obs["teacher_mapping"] if feed == "teacher_mapping" else neural_to_teacher_map(obs["neural_map"])
    if policy == "student":
        return teacher_to_neural_map(obs["teacher_mapping"]) if feed == "teacher_mapping" else obs["neural_map"]
    raise ValueError(policy)


def terrain_column_names(sub_terrains: dict, num_cols: int) -> list[str]:
    """Column index -> sub-terrain name, same rule as the goal-reaching curriculum logger."""
    names = list(sub_terrains.keys())
    props = np.array([float(getattr(s, "proportion", s)) for s in sub_terrains.values()], dtype=np.float64)
    props /= props.sum()
    cum = np.cumsum(props)
    return [names[int(np.min(np.where(i / num_cols + 0.001 < cum)[0]))] for i in range(num_cols)]


class MapErrorTracker:
    """Layer 3: per-step map-vs-GT error per env, per terrain, frame-to-frame jumps and yaw turns."""

    def __init__(self, env_terrain: list[str], yaw_rate_thr: float = 0.5, jump_thr: float = 0.05):
        self.env_terrain = list(env_terrain)
        self.n_env = len(env_terrain)
        self.yaw_thr = yaw_rate_thr
        self.jump_thr = jump_thr
        self.rows = []                       # per-step aggregate rows (CSV)
        self.env_sum = np.zeros(self.n_env)
        self.env_n = np.zeros(self.n_env)
        self.terrain = defaultdict(lambda: defaultdict(float))
        self.prev = None
        self.turn = defaultdict(float)

    @torch.no_grad()
    def update(self, step: int, pred_z, gt_z, observed, yaw_rate, reset=None, gt_valid=None):
        """pred_z, gt_z: (E,L,W) in the same frame; observed: (E,L,W) bool (map has data);
        yaw_rate: (E,) rad/s; reset: (E,) bool envs reset since the last update;
        gt_valid: (E,L,W) bool cells with a real GT hit (others are ignored)."""
        pred_z, gt_z = pred_z.float(), gt_z.float()
        valid = torch.ones_like(gt_z, dtype=torch.bool) if gt_valid is None else gt_valid.bool()
        e = (pred_z - gt_z).abs() * valid
        obs = observed.bool() & valid
        n_obs = obs.float().sum((1, 2))
        n_val = valid.float().sum((1, 2))
        mae_obs = torch.where(n_obs > 0, (e * obs).sum((1, 2)) / n_obs.clamp_min(1), torch.full_like(n_obs, float("nan")))
        mae_all = torch.where(n_val > 0, e.sum((1, 2)) / n_val.clamp_min(1), torch.zeros_like(n_val))
        cov = n_obs / n_val.clamp_min(1)
        jump = torch.full_like(mae_all, float("nan"))
        if self.prev is not None:
            jump = (mae_all - self.prev).abs()
            if reset is not None:
                jump = torch.where(reset.bool(), torch.full_like(jump, float("nan")), jump)
        self.prev = mae_all.clone()
        turning = yaw_rate.abs() > self.yaw_thr
        mae_all_np, mae_obs_np, cov_np = mae_all.cpu().numpy(), mae_obs.cpu().numpy(), cov.cpu().numpy()
        jump_np, turn_np = jump.cpu().numpy(), turning.cpu().numpy()
        self.env_sum += mae_all_np
        self.env_n += 1
        for i, t in enumerate(self.env_terrain):
            d = self.terrain[t]
            d["mae_all_sum"] += mae_all_np[i]
            d["n"] += 1
            if not np.isnan(mae_obs_np[i]):
                d["mae_obs_sum"] += mae_obs_np[i]
                d["n_obs"] += 1
            d["coverage_sum"] += cov_np[i]
            if not np.isnan(jump_np[i]):
                d["jump_sum"] += jump_np[i]
                d["n_jump"] += 1
                d["jumps_over_thr"] += float(jump_np[i] > self.jump_thr)
        for flag, key in ((turn_np, "turning"), (~turn_np, "straight")):
            if flag.any():
                self.turn[key + "_sum"] += float(mae_all_np[flag].sum())
                self.turn[key + "_n"] += float(flag.sum())
                j = jump_np[flag]
                j = j[~np.isnan(j)]
                self.turn[key + "_jump_sum"] += float(j.sum())
                self.turn[key + "_jump_n"] += float(j.size)
        self.rows.append({
            "step": step, "mae_all": float(np.nanmean(mae_all_np)), "mae_observed": float(np.nanmean(mae_obs_np))
            if not np.all(np.isnan(mae_obs_np)) else float("nan"),
            "coverage": float(cov_np.mean()),
            "jump": float(np.nanmean(jump_np)) if not np.all(np.isnan(jump_np)) else float("nan"),
            "turning_frac": float(turn_np.mean()),
        })

    def summary(self) -> dict:
        def r(a, b):
            return a / b if b > 0 else float("nan")
        terr = {t: {"mae_all": r(d["mae_all_sum"], d["n"]), "mae_observed": r(d["mae_obs_sum"], d["n_obs"]),
                    "coverage": r(d["coverage_sum"], d["n"]), "mean_jump": r(d["jump_sum"], d["n_jump"]),
                    "jump_rate": r(d["jumps_over_thr"], d["n_jump"]), "samples": d["n"]}
                for t, d in self.terrain.items()}
        tot_n = sum(d["n"] for d in self.terrain.values())
        return {
            "mae_all": r(sum(d["mae_all_sum"] for d in self.terrain.values()), tot_n),
            "mae_observed": r(sum(d["mae_obs_sum"] for d in self.terrain.values()),
                              sum(d["n_obs"] for d in self.terrain.values())),
            "coverage": r(sum(d["coverage_sum"] for d in self.terrain.values()), tot_n),
            "per_terrain": terr,
            "per_env_mae": (self.env_sum / np.maximum(self.env_n, 1)).tolist(),
            "yaw": {"turning_mae": r(self.turn["turning_sum"], self.turn["turning_n"]),
                    "straight_mae": r(self.turn["straight_sum"], self.turn["straight_n"]),
                    "turning_jump": r(self.turn["turning_jump_sum"], self.turn["turning_jump_n"]),
                    "straight_jump": r(self.turn["straight_jump_sum"], self.turn["straight_jump_n"]),
                    "yaw_rate_thr": self.yaw_thr},
            "jump_thr": self.jump_thr,
        }


class EpisodeOutcomeAggregator:
    """Layer 4: per-terrain success / fall rates over finished episodes.

    success: the episode ran to the time limit and ended within ``success_dist`` of the goal
             (the goal-reaching curriculum's criterion); fall: terminated by any non-time-out term.
    """

    FOCUS = ("stones", "gap", "beam", "pallets")

    def __init__(self, env_terrain: list[str], success_dist: float = 0.5, max_episodes_per_env: int | None = None):
        self.env_terrain = list(env_terrain)
        self.success_dist = success_dist
        self.max_ep = max_episodes_per_env
        self.count = np.zeros(len(env_terrain), dtype=np.int64)
        self.stats = defaultdict(lambda: {"episodes": 0, "success": 0, "fall": 0, "timeout_far": 0})

    def update(self, done, time_out, last_goal_dist):
        done = torch.as_tensor(done).bool().cpu().numpy()
        to = torch.as_tensor(time_out).bool().cpu().numpy()
        dist = torch.as_tensor(last_goal_dist).float().cpu().numpy()
        for i in np.nonzero(done)[0]:
            if self.max_ep is not None and self.count[i] >= self.max_ep:
                continue
            self.count[i] += 1
            s = self.stats[self.env_terrain[i]]
            s["episodes"] += 1
            if not to[i]:
                s["fall"] += 1
            elif dist[i] < self.success_dist:
                s["success"] += 1
            else:
                s["timeout_far"] += 1

    def finished(self) -> bool:
        return self.max_ep is not None and bool((self.count >= self.max_ep).all())

    def summary(self) -> dict:
        per = {}
        for t, s in sorted(self.stats.items()):
            n = max(s["episodes"], 1)
            per[t] = dict(s, success_rate=s["success"] / n, fall_rate=s["fall"] / n)
        tot = {k: sum(s[k] for s in self.stats.values()) for k in ("episodes", "success", "fall", "timeout_far")}
        n = max(tot["episodes"], 1)
        tot.update(success_rate=tot["success"] / n, fall_rate=tot["fall"] / n)
        return {"overall": tot, "per_terrain": per, "focus_terrains": [t for t in self.FOCUS if t in per]}


def compare_modes_markdown(results: dict[str, dict], focus=EpisodeOutcomeAggregator.FOCUS) -> str:
    """results: mode -> EpisodeOutcomeAggregator.summary(). Markdown table, focus terrains first."""
    terrains = sorted({t for r in results.values() for t in r["per_terrain"]},
                      key=lambda t: (t not in focus, t))
    modes = sorted(results)
    head = "| terrain | " + " | ".join(f"{m} success | {m} fall" for m in modes) + " |"
    sep = "|---|" + "---|---|" * len(modes)
    lines = [head, sep]
    for t in terrains + ["**overall**"]:
        cells = []
        for m in modes:
            d = results[m]["overall"] if t == "**overall**" else results[m]["per_terrain"].get(t)
            cells += ["-", "-"] if d is None else [f"{d['success_rate']:.2f} (n={d['episodes']})",
                                                   f"{d['fall_rate']:.2f}"]
        name = f"**{t}**" if t in focus else t
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
