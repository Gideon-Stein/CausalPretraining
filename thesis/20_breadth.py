"""Build tab:cp:synthetic-nauprc (breadth of the training distribution, Ch. 7.1).

    python thesis/ct_baseline.py
    python thesis/rescore.py --mode eval --only single joint_medium joint_big single_cm_fixed joint_cm_fixed
    python thesis/20_breadth.py                 # eval scores (default)
    python thesis/20_breadth.py --mode legacy   # needs rescored_legacy.csv

Reads thesis/results/rescored_<mode>.csv and ct_baseline.csv. Prints the LaTeX tables to stdout
and the number of runs per cell to stderr, and writes one table per metric (common.metrics):
    thesis/outputs/tab_cp_synthetic.tex        nAUPRC (tab:cp:synthetic-nauprc, main metric)
    thesis/outputs/tab_cp_synthetic_nshd.tex   nSHD   (tab:cp:synthetic-nshd)
    thesis/outputs/tab_cp_synthetic_auroc.tex  AUROC  (tab:cp:synthetic-auroc, comparison with the paper only)
    thesis/outputs/tab_cp_joint_subsets.tex  nAUPRC of Single vs Joint big per subset (tab:cp:joint-subsets)
Legacy mode only has the AUROC. Cells are mean (sample std) over reruns; a cell without runs is printed as --.

The Joint test sets are the six Single test sets in blocks of 500 (JOINT_BLOCKS, matched sample by sample against
the Single datasets). Their per-subset scores are computed from the prediction cache and stored in
results/joint_by_subset.csv and results/joint_by_subset_ct.csv (delete them to recompute).
CM comes from the reruns with the corrected head (single_cm_fixed, joint_cm_fixed in manifest/runs.yaml);
the original CM reruns are dropped. Until joint_cm_fixed is scored, the Joint CM cells are --.
"""
import argparse
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (ARCHS, DATASETS, SINGLE, THESIS, cached_predict, deterministic_ds, entry_mask, graph_metrics,
                    number, test_set_path)
from helpers.tools import corr_ci
from rescore import root_of

# block of 500 test samples in joint(_additional)/simple_test.p -> Single dataset
JOINT_BLOCKS = {0: "ML", 1: "XLNL", 2: "MNL", 3: "SNL", 4: "LNL", 5: "SL"}

JOINT = {"medium": "Medium", "big": "Big"}
CM = ARCHS["convM"]
# test set -> (title row color, body row color)
TEST_SETS = {1: ("82B366", "eff6ee"), 2: ("D79B00", "fff8e6")}
# (experiment, arch, seed) of runs killed before early stopping, per the callback state in last.ckpt.
# They have a checkpoint but no finished training; the other 398 runs all stopped regularly.
UNFINISHED = [("joint_big", "bGRU", 43), ("joint_big", "bGRU", 44)]

# metric -> (file suffix, label, CT column in ct_baseline.csv, caption start)
METRICS = {
    "nauprc": ("", "tab:cp:synthetic-nauprc", "nauprc", r"Mean nAUPRC (per graph, $0$ = random ranking, $1$ = perfect)"),
    "nshd": ("_nshd", "tab:cp:synthetic-nshd", "nshd", r"Mean nSHD (per graph, lower is better)"),
    "auroc": ("_auroc", "tab:cp:synthetic-auroc", "auroc_scored", r"Mean AUROC (pooled over all graphs)"),
}
CAPTION = (r" for the \textbf{Single} and \textbf{Joint} settings, with the standard deviation "
           r"over reruns in brackets. Graphs without edges are excluded, and three-variable samples of "
           r"\textbf{Joint} are scored at their true size. For \textbf{Single}, the second column names the dataset; for "
           r"\textbf{Joint}, it names the model size. All cells are based on 10 reruns, except for bGRU in "
           r"\textbf{Joint} \emph{big}, where two reruns did not finish training (8 reruns). CM uses the corrected "
           r"output head (\cref{cp:architectures}). CT is the correlation-thresholding reference.")


def cells(runs, ts, metric):
    """One mean (std) cell per architecture."""
    out = []
    for arch in ARCHS.values():
        v = runs.loc[runs["arch"] == arch, f"ts{ts}_{metric}"]
        out.append(f"{number(v.mean())} ({number(v.std())})" if len(v) else "--")
    return " & ".join(out)


def single_runs(df, name):
    return df[(df["experiment"] == "single") & (df["dataset"] == DATASETS[name])]


def joint_runs(df, size):
    return df[df["experiment"] == f"joint_{size}"]


def ct_score(ct, name, ts, metric):
    row = ct[(ct["experiment"] == "single_joint") & (ct["dataset"] == DATASETS[name]) & (ct["split"] == f"ts{ts}")]
    return number(row[METRICS[metric][2]].item())


def test_set_rows(df, ct, ts, metric):
    title_color, color = TEST_SETS[ts]
    rowcolor = rf"\rowcolor[HTML]{{{color}}}"
    lines = [rf"\rowcolor[HTML]{{{title_color}}}", rf"\multicolumn{{8}}{{c}}{{\textit{{Test-Set {ts}}}}}\\", r"\midrule"]
    for name in SINGLE:
        first = r"\multirow{-6}{*}{\rotatebox[origin=c]{90}{\textbf{Single}}}" if name == SINGLE[-1] else "~"
        lines.append(f"{rowcolor}{first} & \\textbf{{{name}}} & {cells(single_runs(df, name), ts, metric)} "
                     f"& {ct_score(ct, name, ts, metric)} \\\\")
    lines += [r"\midrule\midrule",
              f"{rowcolor}~ & \\textbf{{{JOINT['medium']}}} & {cells(joint_runs(df, 'medium'), ts, metric)} & \\\\",
              f"{rowcolor}\\multirow{{-2}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{Joint}}}}}} "
              f"& \\textbf{{{JOINT['big']}}} & {cells(joint_runs(df, 'big'), ts, metric)} "
              f"& \\multirow{{-2}}{{*}}{{{ct_score(ct, 'Joint', ts, metric)}}} \\\\"]
    return lines


def latex_table(df, ct, metric):
    header = " & ".join(rf"\textbf{{{a}}}" for a in ARCHS.values())
    lines = [r"\begin{table*}[ht]", r"    \centering", r"    \begin{tabular}{l|lccccc|c}", r"    \toprule",
             rf"        ~ & ~& {header}& \textbf{{CT}}\\ \hline"]
    rows = test_set_rows(df, ct, 1, metric) + [r"\midrule"] + test_set_rows(df, ct, 2, metric)
    lines += ["        " + r for r in rows]
    lines += [r"    \bottomrule", r"    \end{tabular}", rf"    \caption{{{METRICS[metric][3] + CAPTION}}}",
              rf"\label{{{METRICS[metric][1]}}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def run_counts(df):
    """Runs per table cell, to spot incomplete or still missing experiments."""
    parts = {name: single_runs(df, name) for name in SINGLE}
    parts.update({label: joint_runs(df, size) for size, label in JOINT.items()})
    counts = {k: v.groupby("arch").size() for k, v in parts.items()}
    return pd.DataFrame(counts).T.reindex(columns=list(ARCHS.values())).fillna(0).astype(int)


def with_cm_fixed(df):
    """Replace the original CM reruns of Single and Joint by the reruns with the corrected head."""
    table = df["experiment"].isin(["single", "joint_medium", "joint_big"])
    fixed = df[df["experiment"].isin(["single_cm_fixed", "joint_cm_fixed"])].copy()
    is_joint = fixed["experiment"] == "joint_cm_fixed"
    fixed.loc[~is_joint, "experiment"] = "single"
    fixed.loc[is_joint, "experiment"] = "joint_" + fixed.loc[is_joint, "size"]
    return pd.concat([df[table & (df["arch"] != CM)], fixed], ignore_index=True)


def _by_block(g):
    return [(name, g.iloc[b * 500:(b + 1) * 500]) for b, name in JOINT_BLOCKS.items()]


def joint_subsets(df):
    """nAUPRC and nSHD of every Joint rerun (and of CT) per Single subset of the Joint test sets (cached)."""
    out, out_ct = THESIS / "results" / "joint_by_subset.csv", THESIS / "results" / "joint_by_subset_ct.csv"
    if not out.exists():
        rows = []
        for _, r in df[df["experiment"].str.startswith("joint")].iterrows():
            run = root_of("joint_cm_fixed" if r["arch"] == CM else r["experiment"]) / r["run_dir"]
            for ts in (1, 2):
                path = test_set_path("Joint", ts)
                pred, lab = cached_predict(run, path)
                for name, g in _by_block(graph_metrics(pred, lab, entry_mask(path, lab.shape))):
                    rows.append({"size": r["size"], "arch": r["arch"], "seed": r["seed"], "ts": ts, "subset": name,
                                 "nauprc": g["nauprc"].mean(), "nshd": g["nshd"].mean()})
        pd.DataFrame(rows).to_csv(out, index=False)
    if not out_ct.exists():
        rows = []
        for ts in (1, 2):
            path = test_set_path("Joint", ts)
            ds = deterministic_ds(ds_path=str(path), binary=True, corr_input=False)
            xs, ys = zip(*DataLoader(ds, batch_size=500, shuffle=False))
            pred = torch.cat([corr_ci(x, max_lags=y.shape[3])[0] for x, y in zip(xs, ys)])
            lab = torch.cat(ys)
            for name, g in _by_block(graph_metrics(pred, lab, entry_mask(path, lab.shape))):
                rows.append({"ts": ts, "subset": name, "nauprc": g["nauprc"].mean(), "nshd": g["nshd"].mean()})
        pd.DataFrame(rows).to_csv(out_ct, index=False)
    return pd.read_csv(out), pd.read_csv(out_ct)


def subsets_table(df, joint, ct, ts=2, metric="nauprc"):
    """Rows: Single datasets; columns: each architecture trained on it alone (Single) or in Joint big; CT."""
    archs = list(ARCHS.values())
    j = joint[(joint["ts"] == ts) & (joint["size"] == "big")].groupby(["subset", "arch"])[metric].mean()
    c = ct[ct["ts"] == ts].set_index("subset")[metric]
    lines = [r"\begin{table*}[ht]", r"    \centering", r"    \small", r"    \setlength{\tabcolsep}{4pt}",
             r"    \begin{tabular}{l" + "c" * len(archs) + "|" + "c" * len(archs) + "|c}", r"    \toprule",
             rf"        ~ & \multicolumn{{{len(archs)}}}{{c|}}{{\textbf{{Single}}}} & "
             rf"\multicolumn{{{len(archs)}}}{{c|}}{{\textbf{{Joint}} (\emph{{big}})}} & ~ \\",
             "        ~ & " + " & ".join(rf"\textbf{{{a}}}" for a in archs * 2) + r" & \textbf{CT} \\", r"    \midrule"]
    for name in SINGLE:
        single = single_runs(df, name).groupby("arch")[f"ts{ts}_{metric}"].mean()
        cells = [number(single[a]) for a in archs] + [number(j[(name, a)]) for a in archs]
        lines.append(rf"        \textbf{{{name}}} & " + " & ".join(cells) + f" & {number(c[name])} \\\\")
    caption = (rf"Mean nAUPRC on \textit{{Test-Set {ts}}} per dataset: models trained on the dataset alone (\textbf{{Single}}, "
               r"size \emph{small} for three and \emph{medium} for five variables) and models of size \emph{big} "
               r"trained on all six datasets (\textbf{Joint}), scored on the samples of the same dataset within the "
               r"\textbf{Joint} test set. Means over the reruns of \cref{tab:cp:synthetic-nauprc}.")
    lines += [r"    \bottomrule", r"    \end{tabular}", rf"    \caption{{{caption}}}", r"\label{tab:cp:joint-subsets}",
              r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["eval", "legacy"], default="eval")
    args = parser.parse_args()

    df = pd.read_csv(THESIS / "results" / f"rescored_{args.mode}.csv")
    keys = pd.MultiIndex.from_frame(df[["experiment", "arch", "seed"]])
    df = with_cm_fixed(df[~keys.isin(UNFINISHED)])
    ct = pd.read_csv(THESIS / "results" / "ct_baseline.csv")
    if args.mode == "legacy":  # legacy CP scores include the Joint padding, as does the stored CT AUROC
        METRICS["auroc"] = ("_auroc", "tab:cp:synthetic-auroc", "recomputed", METRICS["auroc"][3])
    (THESIS / "outputs").mkdir(exist_ok=True)
    for metric, (suffix, *_ ) in METRICS.items():
        if f"ts1_{metric}" not in df:
            continue
        table = latex_table(df, ct, metric)
        (THESIS / "outputs" / f"tab_cp_synthetic{suffix}.tex").write_text(table)
        print(table)
    if args.mode == "eval":
        joint, joint_ct = joint_subsets(df)
        table = subsets_table(df, joint, joint_ct)
        (THESIS / "outputs" / "tab_cp_joint_subsets.tex").write_text(table)
        print(table)
    print(f"runs per cell ({args.mode}):\n{run_counts(df).to_string()}", file=sys.stderr)


if __name__ == "__main__":
    main()
