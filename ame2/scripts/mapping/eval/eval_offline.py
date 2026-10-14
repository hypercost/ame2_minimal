"""Layer 1b: evaluate one or more mapping checkpoints on a frozen test set.

Per family and overall: observed-cell MAE vs. the raw input, hole MAE vs. mean-fill and
nearest-fill (onion-peel inpaint) baselines bucketed by distance to the nearest observed
cell (1-3 / 4-8 / >8 cells), calibration (within 1 / 2 sigma), sparsification + AUSE,
gate statistics. Writes summary.json, summary.md and PNG plots to --out_dir.

    python ame2/scripts/mapping/eval/eval_offline.py --testset data/mapping/testset_seed777.pt \\
        --checkpoints logs/mapping/mapping_ckpt.pt ame2/ame2/NN_models/g1LivoxMapping_perframe.pt dummy \\
        --out_dir logs/mapping_eval/offline
"""

from __future__ import annotations

import argparse
import json
import math
import os

import _common
import torch

from ame2.mapping.eval_metrics import BUCKET_NAMES, OfflineAccumulator

# pass criteria (see README "Mapping-model validation")
CRITERIA = {
    "observed_mae_le_raw": "observed-cell MAE <= raw-input MAE",
    "hole_mae_beats_baselines": "hole MAE < min(mean-fill, nearest-fill), overall and in every distance bucket",
    "calibration": "within-1sigma in [0.55, 0.85] and within-2sigma in [0.85, 0.99]",
    "ause": "AUSE < 0.5 * AUSE of a random uncertainty ranking (sigma ranks errors usefully)",
}


def judge(s: dict) -> dict:
    ok = {}
    ok["observed_mae_le_raw"] = s["observed_mae"]["model"] <= s["observed_mae"]["raw_input"] + 1e-6
    hb = s["hole_mae"]
    good = hb["model"] < min(hb["mean_fill"], hb["nearest_fill"])
    for b in BUCKET_NAMES:
        d = s["hole_mae_by_distance"][b]
        if d["cells"] > 0:
            good &= d["model"] < min(d["mean_fill"], d["nearest_fill"])
    ok["hole_mae_beats_baselines"] = bool(good)
    ok["calibration"] = 0.55 <= s["within_1sigma"] <= 0.85 and 0.85 <= s["within_2sigma"] <= 0.99
    ok["ause"] = s.get("ause", float("inf")) < 0.5 * s.get("ause_random", 0.0)
    return {k: bool(v) for k, v in ok.items()}


def evaluate(model, data, device, batch: int):
    fams = data["families"]
    hole = data["spec"]["hole_value"]
    accs = {"overall": OfflineAccumulator(hole)} | {f: OfflineAccumulator(hole) for f in fams}
    n = data["x"].shape[0]
    for i in range(0, n, batch):
        x = data["x"][i:i + batch].float().unsqueeze(1).to(device)
        y = data["y"][i:i + batch].float().unsqueeze(1).to(device)
        m = data["mask"][i:i + batch].unsqueeze(1).to(device)
        f = data["family"][i:i + batch].to(device)
        mu, sigma, gate = _common.predict(model, x)
        accs["overall"].update(x, y, m, mu, sigma, gate)
        for j, name in enumerate(fams):
            sel = f == j
            if sel.any():
                accs[name].update(x[sel], y[sel], m[sel], mu[sel], sigma[sel], None if gate is None else gate[sel])
    return {k: a.summary() for k, a in accs.items()}


def _fmt(v):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def markdown(results: dict) -> str:
    lines = ["# Mapping model offline evaluation", ""]
    for name, res in results.items():
        lines += [f"## {name}", "", "| split | obs MAE (raw) | hole MAE (mean / nearest) | "
                  + " | ".join(f"hole {b} (mean / nearest)" for b in BUCKET_NAMES)
                  + " | in 1σ | in 2σ | AUSE (random) | gate clean / corrupted |", "|---" * 10 + "|"]
        for split, s in res["splits"].items():
            g = s.get("gate")
            row = [split, f"{_fmt(s['observed_mae']['model'])} ({_fmt(s['observed_mae']['raw_input'])})",
                   f"{_fmt(s['hole_mae']['model'])} ({_fmt(s['hole_mae']['mean_fill'])} / {_fmt(s['hole_mae']['nearest_fill'])})"]
            for b in BUCKET_NAMES:
                d = s["hole_mae_by_distance"][b]
                row.append(f"{_fmt(d['model'])} ({_fmt(d['mean_fill'])} / {_fmt(d['nearest_fill'])})")
            row += [_fmt(s["within_1sigma"]), _fmt(s["within_2sigma"]), f"{_fmt(s.get('ause'))} ({_fmt(s.get('ause_random'))})",
                    "-" if g is None else f"{_fmt(g['observed_clean'])} / {_fmt(g['observed_corrupted'])}"]
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "Pass criteria (overall): " + ", ".join(
            f"{'PASS' if v else 'FAIL'} {CRITERIA[k]}" for k, v in res["pass"].items()), ""]
    return "\n".join(lines)


def plots(results: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for name, res in results.items():
        c = res["splits"]["overall"].get("calibration")
        if c:
            ax.plot(c["expected"], c["observed"], "o-", label=name)
    ax.set_xlabel("expected coverage (Gaussian)"), ax.set_ylabel("observed coverage |err| <= k sigma")
    ax.set_title("Calibration"), ax.legend(fontsize=7), ax.grid(True)
    paths.append(os.path.join(out_dir, "calibration.png")), fig.savefig(paths[-1], dpi=120), plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, res) in enumerate(results.items()):
        sp = res["splits"]["overall"].get("sparsification")
        if sp:
            ax.plot(sp["fractions"], sp["model"], "-", color=f"C{i}", label=f"{name} (AUSE {sp['ause']:.4f})")
            ax.plot(sp["fractions"], sp["oracle"], ":", color=f"C{i}")
    ax.set_xlabel("fraction of most-uncertain cells removed"), ax.set_ylabel("MAE of remaining [m]")
    ax.set_title("Sparsification (dotted = oracle)"), ax.legend(fontsize=7), ax.grid(True)
    paths.append(os.path.join(out_dir, "sparsification.png")), fig.savefig(paths[-1], dpi=120), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / (len(results) + 2)
    xs = range(len(BUCKET_NAMES))
    first = next(iter(results.values()))["splits"]["overall"]["hole_mae_by_distance"]
    for j, (lab, key) in enumerate((("mean-fill", "mean_fill"), ("nearest-fill", "nearest_fill"))):
        ax.bar([x + j * width for x in xs], [first[b][key] for b in BUCKET_NAMES], width, label=lab, hatch="//")
    for i, (name, res) in enumerate(results.items()):
        d = res["splits"]["overall"]["hole_mae_by_distance"]
        ax.bar([x + (i + 2) * width for x in xs], [d[b]["model"] for b in BUCKET_NAMES], width, label=name)
    ax.set_xticks([x + width * (len(results) + 1) / 2 for x in xs], [f"{b} cells" for b in BUCKET_NAMES])
    ax.set_ylabel("hole MAE [m]"), ax.set_title("Error by distance to nearest observed cell")
    ax.legend(fontsize=7), ax.grid(True, axis="y")
    paths.append(os.path.join(out_dir, "hole_error_by_size.png")), fig.savefig(paths[-1], dpi=120), plt.close(fig)
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testset", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True, help="ckpt / TorchScript paths, or 'dummy'")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="logs/mapping_eval/offline")
    a = p.parse_args(argv)
    device = torch.device(a.device)
    data = torch.load(a.testset, map_location="cpu", weights_only=False)
    if a.max_frames:
        for k in ("x", "y", "mask", "family"):
            data[k] = data[k][:a.max_frames]
    os.makedirs(a.out_dir, exist_ok=True)
    results = {}
    for ck in a.checkpoints:
        model = _common.load_model(ck, device)
        splits = evaluate(model, data, device, a.batch)
        results[_common.model_name(ck)] = {"path": ck, "splits": splits, "pass": judge(splits["overall"])}
        s = splits["overall"]
        print(f"[eval] {ck}: obs MAE {s['observed_mae']['model']:.4f} (raw {s['observed_mae']['raw_input']:.4f}), "
              f"hole MAE {s['hole_mae']['model']:.4f} (mean {s['hole_mae']['mean_fill']:.4f}, "
              f"nearest {s['hole_mae']['nearest_fill']:.4f}), 1σ {s['within_1sigma']:.3f}, "
              f"2σ {s['within_2sigma']:.3f}, AUSE {s.get('ause', float('nan')):.4f}", flush=True)
    meta = {"testset": a.testset, "frames": int(data["x"].shape[0]), "families": data["families"],
            "criteria": CRITERIA}
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1, default=float)
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write(markdown(results))
    paths = plots(results, a.out_dir)
    print(f"[eval] wrote {a.out_dir}/summary.json, summary.md, " + ", ".join(os.path.basename(x) for x in paths))
    return results


if __name__ == "__main__":
    main()
