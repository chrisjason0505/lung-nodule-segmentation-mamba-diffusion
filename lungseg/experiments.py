"""Label-efficiency study: {UNet, Mamba-UNet, Mamba-UNet + diffusion pretraining}
x {10, 25, 50, 100}% of labelled training patients, one fixed test set.

    python -m lungseg.experiments --data data/lidc_crops --runs runs
    python -m lungseg.experiments --data data/lidc_crops --runs runs --aggregate_only

Runs whose ``results.json`` already exists are skipped, so you can stop and
restart (for example across Kaggle sessions) without losing work.
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import torch

PRESETS = {
    # name: list of (config_name, arch, pretrained)
    "core": [("unet", "unet", False), ("mamba", "mamba", False), ("mamba+ddep", "mamba", True)],
    "full": [("unet", "unet", False), ("unet+ddep", "unet", True),
             ("mamba", "mamba", False), ("mamba+ddep", "mamba", True)],
    "mamba_only": [("mamba+ddep", "mamba", True)],
}


def run_dir(runs, cfg, frac, seed):
    return os.path.join(runs, f"{cfg.replace('+', '_')}_f{frac:g}_s{seed}")


def aggregate(runs):
    import csv
    import numpy as np
    rows = []
    for d in sorted(os.listdir(runs)):
        p = os.path.join(runs, d, "results.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if "test" not in r:
            continue
        c, t = r["config"], r["test"]["all"]
        rows.append(dict(run=d, config=c.get("name", c["arch"]), arch=c["arch"], pretrained=bool(c["init"]),
                         fraction=c["fraction"], seed=c["seed"], train_patients=c["n_train_patients"],
                         train_nodules=c["n_train"], params_M=round(c["n_params"] / 1e6, 2),
                         test_n=t["n"], dice=t["dice"], dice_std=t["dice_std"], iou=t["iou"],
                         precision=t["precision"], recall=t["recall"], hd95_mm=t.get("hd95_mm", float("nan")),
                         radiologist_dice=r["test"].get("radiologists_leave_one_out", {}).get("dice", float("nan")),
                         val_dice=r["best_val_dice"]))
    if not rows:
        print("no finished runs yet")
        return rows
    with open(os.path.join(runs, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # mean over seeds
    groups = {}
    for r in rows:
        groups.setdefault((r["config"], r["fraction"]), []).append(r)
    lines = ["| config | labelled patients | train nodules | test Dice | test IoU | HD95 (mm) | seeds |",
             "|---|---|---|---|---|---|---|"]
    table = []
    for (cfg, frac), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        d = np.array([x["dice"] for x in g]); i = np.array([x["iou"] for x in g])
        h = np.array([x["hd95_mm"] for x in g])
        table.append((cfg, frac, d.mean(), i.mean()))
        pm = f" ± {d.std():.3f}" if len(g) > 1 else ""
        lines.append(f"| {cfg} | {int(frac * 100)}% ({g[0]['train_patients']}) | {g[0]['train_nodules']} | "
                     f"{d.mean():.4f}{pm} | {i.mean():.4f} | {np.nanmean(h):.2f} | {len(g)} |")
    rad = [r["radiologist_dice"] for r in rows if r["radiologist_dice"] == r["radiologist_dice"]]
    if rad:
        lines.append(f"\nRadiologist leave-one-out Dice on the same test nodules: {rad[0]:.4f}")
    md = "\n".join(lines)
    open(os.path.join(runs, "summary.md"), "w").write(md + "\n")
    print(md)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, lab in ((axes[0], 2, "test Dice"), (axes[1], 3, "test IoU")):
            for cfg in sorted({t[0] for t in table}):
                pts = sorted((t[1], t[key]) for t in table if t[0] == cfg)
                ax.plot([p[0] * 100 for p in pts], [p[1] for p in pts], marker="o", label=cfg)
            if key == 2 and rad:
                ax.axhline(rad[0], ls="--", c="gray", lw=1, label="radiologists (leave-one-out)")
            ax.set_xscale("log")
            ax.set_xticks([10, 25, 50, 100])
            ax.set_xticklabels(["10%", "25%", "50%", "100%"])
            ax.set_xlabel("labelled training patients")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(runs, "data_efficiency.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("plot failed:", e)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--preset", default="core", choices=list(PRESETS))
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--pretrain_steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--configs", nargs="+", default=None, help="subset of the preset's config names")
    ap.add_argument("--deadline", type=float, default=0,
                    help="unix time by which everything must be finished; each run gets an equal share")
    ap.add_argument("--extra", default="", help="extra args passed to train/pretrain, e.g. '--widths 24 48 96 192 256'")
    args = ap.parse_args(argv)
    os.makedirs(args.runs, exist_ok=True)
    if args.aggregate_only:
        aggregate(args.runs)
        return

    from lungseg import pretrain, train
    extra = args.extra.split() if args.extra else []
    import time
    configs = PRESETS[args.preset]
    if args.configs:
        configs = [c for c in configs if c[0] in args.configs]
    todo = [(n, a, p, f, s) for f in args.fractions for s in args.seeds for n, a, p in configs
            if not os.path.exists(os.path.join(run_dir(args.runs, n, f, s), "results.json"))]
    pre_todo = [a for a in sorted({a for _, a, pre in configs if pre})
                if not os.path.exists(os.path.join(args.runs, f"pretrain_{a}", "pretrain.pt"))]
    n_left = [len(todo) + len(pre_todo)]

    def budget():
        if not args.deadline:
            return []
        mins = max((args.deadline - time.time()) / 60 - 5, 5) / max(n_left[0], 1)
        n_left[0] -= 1
        return ["--max_minutes", f"{mins:.1f}"]

    for arch in pre_todo:
        out = os.path.join(args.runs, f"pretrain_{arch}")
        if True:
            print(f"\n=== denoising pretraining: {arch} ===", flush=True)
            pretrain.main(["--data", args.data, "--out", out, "--arch", arch, "--steps", str(args.pretrain_steps),
                           "--batch", str(args.batch)] + extra + budget())
            gc.collect(); torch.cuda.empty_cache()
    for frac in args.fractions:
        for seed in args.seeds:
            for name, arch, pre in configs:
                out = run_dir(args.runs, name, frac, seed)
                if os.path.exists(os.path.join(out, "results.json")):
                    print(f"[skip] {out}")
                    continue
                print(f"\n=== {name}  fraction {frac}  seed {seed} ===", flush=True)
                argv_t = ["--data", args.data, "--out", out, "--arch", arch, "--fraction", str(frac),
                          "--seed", str(seed), "--steps", str(args.steps), "--batch", str(args.batch)] + extra + budget()
                if pre:
                    argv_t += ["--init", os.path.join(args.runs, f"pretrain_{arch}", "pretrain.pt")]
                res = train.main(argv_t)
                res["config"]["name"] = name
                with open(os.path.join(out, "results.json"), "w") as f:
                    json.dump(res, f, indent=1)
                gc.collect(); torch.cuda.empty_cache()
                aggregate(args.runs)
    aggregate(args.runs)


if __name__ == "__main__":
    main()
