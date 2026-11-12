"""Build fig:cp:scaling (scaling with model size, Ch. 7.1).

    python thesis/rescore.py --mode eval --only scaling scaling_deep
    python thesis/30_scaling.py                 # Test-Set 2 (default)
    python thesis/30_scaling.py --test-set 1

For every combination of number of variables and model size, the architecture with the lowest
mean validation MSE is selected, and its mean and sample std over reruns of nAUPRC and nSHD (per graph,
common.metrics) and the pooled AUROC is reported on the test set; the figure shows nAUPRC and nSHD. val_loss is not used, because it includes the correlation regularization in some runs.
LCM is restricted to CM and Trf (as in tab:cp:params); 12 variables have no runs.

Validation scores are computed once from the checkpoints and cached per run in
thesis/results/scaling_val.csv. Writes thesis/results/scaling_summary.csv and
thesis/outputs/fig_cp_scaling.{pdf,png}. The summary is also printed, including the 15-variable
dataset, which is not plotted because every size fails on it.
"""
import argparse

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (PATHS, SCALE_UP, THESIS, auroc, best_checkpoint, load_model, predict,  # noqa: E402
                    run_config)
from rescore import runs_of  # noqa: E402

SIZES = {"small": "Small", "medium": "Medium", "big": "Big", "deep": "Deep", "lcm": "LCM"}
LCM_ARCHS = ["CM", "Trf"]
PLOTTED_VARS = [5, 7, 10]
N_VARS = {folder: n for n, folder in SCALE_UP.items()}

# one hue, light to dark, in the order of the sizes
COLORS = dict(zip(SIZES, ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]))
MARKERS = dict(zip(SIZES, ["o", "s", "^", "D", "X"]))
DODGE = 0.1
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def validation_scores():
    """Validation MSE and AUROC of every scaling run, in eval mode (cached, resumes)."""
    cache = THESIS / "results" / "scaling_val.csv"
    cached = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["run_dir", "val_mse", "val_auroc"])
    done = set(cached["run_dir"])
    seen = set()
    for name in ("scaling", "scaling_deep"):
        for run_dir, ov in runs_of(name):
            key = str(run_dir.relative_to(PATHS["runs_root"]))
            ckpt = best_checkpoint(run_dir)
            if key in done or key in seen or ckpt is None:
                continue
            seen.add(key)
            path = PATHS["data_root"] / ov["data.ds_name"] / "simple_val.p"
            pred, lab = predict(load_model(ckpt), path, binary=run_config(run_dir)["data"]["binary"])
            row = {"run_dir": key, "val_mse": torch.mean((pred - lab) ** 2).item(), "val_auroc": auroc(pred, lab)}
            print(f"[val] {key} MSE={row['val_mse']:.4f} AUROC={row['val_auroc']:.4f}", flush=True)
            pd.DataFrame([row]).to_csv(cache, mode="a", header=not cache.exists(), index=False)
    return pd.read_csv(cache)


def summarize(df, val):
    """One row per (variables, size): the selected architecture and its scores over reruns."""
    df = df[df["experiment"].isin(["scaling", "scaling_deep"])].merge(val, on="run_dir")
    df = df[(df["size"] != "lcm") | df["arch"].isin(LCM_ARCHS)]
    df = df.assign(n_vars=df["dataset"].map(N_VARS))
    rows = []
    for (n_vars, size), group in df.groupby(["n_vars", "size"]):
        arch = group.groupby("arch")["val_mse"].mean().idxmin()
        runs = group[group["arch"] == arch]
        row = {"n_vars": n_vars, "size": size, "arch": arch, "reruns": len(runs), "val_mse": runs["val_mse"].mean()}
        for ts in (1, 2):
            for m in ("nauprc", "nshd", "auroc"):
                row[f"ts{ts}_{m}_mean"] = runs[f"ts{ts}_{m}"].mean()
                row[f"ts{ts}_{m}_std"] = runs[f"ts{ts}_{m}"].std()
        rows.append(row)
    order = {s: i for i, s in enumerate(SIZES)}
    return pd.DataFrame(rows).sort_values(["n_vars", "size"], key=lambda c: c.map(order) if c.name == "size" else c)


# metric -> (axis label, y limits); nAUPRC: 0 = random ranking; nSHD: lower is better, 1 = empty graph
PANELS = {"nauprc": ("nAUPRC", (-0.05, 1.02)), "nshd": ("nSHD (lower is better)", (-0.02, 1.05))}


def plot(summary, ts, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharex=True)
    for ax, (metric, (ylabel, ylim)) in zip(axes, PANELS.items()):
        for i, (size, label) in enumerate(SIZES.items()):
            s = summary[(summary["size"] == size) & summary["n_vars"].isin(PLOTTED_VARS)]
            if s.empty:
                continue
            # sizes are dodged around the tick so that their error bars stay readable
            ax.errorbar(s["n_vars"] + DODGE * (i - 2), s[f"ts{ts}_{metric}_mean"], yerr=s[f"ts{ts}_{metric}_std"],
                        label=label, color=COLORS[size], marker=MARKERS[size], ms=6, lw=2, elinewidth=1,
                        capsize=2.5, zorder=3)
        ax.set_xticks(PLOTTED_VARS)
        ax.set_xlim(PLOTTED_VARS[0] - 0.6, PLOTTED_VARS[-1] + 0.6)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Number of variables", color=INK)
        ax.set_ylabel(f"{ylabel}, Test-Set {ts}", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[0].axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    axes[0].text(PLOTTED_VARS[0] - 0.5, -0.045, "random", color=MUTED, fontsize=9, ha="left", va="bottom")
    axes[1].legend(title="Model size", frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(f"{out}.{suffix}", dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["eval", "legacy"], default="eval")
    parser.add_argument("--test-set", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()

    df = pd.read_csv(THESIS / "results" / f"rescored_{args.mode}.csv")
    summary = summarize(df, validation_scores())
    summary.to_csv(THESIS / "results" / "scaling_summary.csv", index=False)
    (THESIS / "outputs").mkdir(exist_ok=True)
    plot(summary, args.test_set, THESIS / "outputs" / "fig_cp_scaling")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
