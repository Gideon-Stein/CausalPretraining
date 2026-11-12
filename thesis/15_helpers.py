"""Effect of correlation injection, the edge-count head and CR on the Single grid (Ch. 7.1, App.).

    python thesis/rescore_grid.py      # once: eval-mode scores of all grid runs
    python thesis/15_helpers.py [--split val|ts1|ts2] [--metric nauprc|nshd|auroc]

The Single grid is a full factorial design per dataset: 5 architectures x lr {1e-4, 1e-3} x batch size
{16, 64, 256} x weight decay {0, 0.01} x the 2^3 on/off combinations of correlation injection (CI),
the edge-count regression head and correlation regularization (CR). The two passes of the grid use
the same seed, so they are averaged per configuration (a configuration with only one completed pass
keeps that one). Model size goes with the dataset (3 variables: small, 5 variables: medium), so all
results are split by the number of variables.

For each mechanism, architecture and number of variables, over the datasets and all other hyperparameters:
paired     mean metric difference (on - off) between configurations that differ only in this mechanism
converged  share of configurations that reach the CONVERGED threshold of the metric with the mechanism off -> on
best       mean over datasets of the best score with the mechanism off -> on
The default metric is nAUPRC (per graph, common.metrics); for nSHD, lower is better, so converged means at
most the threshold and best is the minimum.

The default split is the validation split (coefficients from [0.2, 0.3], training noise), on which
the grid was originally evaluated. Writes
    thesis/outputs/tab_cp_ci<suffix>.tex       (tab:cp:ci, main text: CI)
    thesis/outputs/tab_cp_helpers<suffix>.tex  (tab:cp:helpers, appendix: edge-count head and CR)
    thesis/results/helpers_<split><suffix>.csv (all effects, also per dataset and split by CI)
with suffix "" for nAUPRC on the validation split, else _<split> and/or _<metric>.
"""
import argparse

import pandas as pd

from common import ARCHS, THESIS, number

MECHANISMS = {"corr_input": "Correlation injection", "regression_head": "Edge-count head",
              "model.corr_regularization": "Correlation regularization (CR)"}
OTHER = ["dataset", "arch", "model.optimizer_lr", "data.batch_size", "model.weight_decay"]
# metric -> (label, converged threshold, higher is better)
METRICS = {"nauprc": ("nAUPRC", 0.1, True), "nshd": ("nSHD", 0.9, False), "auroc": ("AUROC", 0.55, True)}
METRIC = "nauprc"  # set in main()
N_VARS = {"base": 3, "poly": 3, "nl": 3, "big_base_mirror": 5, "big_poly": 5, "big_nl": 5}
WORDS = {1: "one", 2: "two", 3: "three"}
SPLIT_TEXT = {"val": "the validation split (coefficients from $[0.2,0.3]$)",
              "ts1": r"\textit{Test-Set 1}", "ts2": r"\textit{Test-Set 2}"}


def configurations(split):
    """Grid configurations; CM from the grid with the corrected head (rescore_grid.py --cm-fixed, one pass)."""
    df = pd.read_csv(THESIS / "results" / "grid_eval.csv")
    cm = pd.read_csv(THESIS / "results" / "grid_eval_cm_fixed.csv")
    df = pd.concat([df[df["arch"] != "CM"], cm[df.columns.intersection(cm.columns)]], ignore_index=True)
    for m in MECHANISMS:
        df[m] = df[m].astype(str).map({"True": True, "False": False})
    keys = OTHER + list(MECHANISMS)
    col = f"{split}_{METRIC}"
    df = df.groupby(keys, as_index=False)[col].mean().rename(columns={col: "score"})
    df["n_vars"] = df["dataset"].map(N_VARS)
    return df


def pairs(df, mechanism):
    """One row per pair of configurations that differ only in `mechanism`: score off, on."""
    keys = [k for k in OTHER + list(MECHANISMS) if k != mechanism]
    wide = df.pivot_table(index=keys, columns=mechanism, values="score").dropna()
    return wide.rename(columns={False: "off", True: "on"}).reset_index()


def effects(df, mechanism, by="arch"):
    _, threshold, higher = METRICS[METRIC]
    conv = (lambda v: v >= threshold) if higher else (lambda v: v <= threshold)
    p = pairs(df, mechanism)
    rows = []
    for name, g in list(p.groupby(by)) + [("All", p)]:
        sub = df[df[by] == name] if name != "All" else df
        grouped = sub.groupby(["dataset", mechanism])["score"]
        best = (grouped.max() if higher else grouped.min()).unstack().dropna()  # datasets with both
        rows.append({by: name, "pairs": len(g), "paired": (g["on"] - g["off"]).mean(),
                     "conv_off": conv(g["off"]).mean(), "conv_on": conv(g["on"]).mean(),
                     "best_off": best[False].mean(), "best_on": best[True].mean()})
    return pd.DataFrame(rows)


def signed(x):
    s = number(abs(x))
    return ("$-$" if x < 0 else "+") + s if round(abs(x), 3) > 0 else s


def arrow(off, on, fmt):
    return rf"{fmt(off)}\,$\to$\,{fmt(on)}"


def percent(x):
    return str(round(100 * x))


def short(x):
    """Two decimals without the leading zero, for the off -> on cells."""
    s = f"{x:.2f}"
    return "1.00" if s == "1.00" else s.removeprefix("0")


def block(df, mechanism, archs):
    label = METRICS[METRIC][0]
    lines = []
    for n in (3, 5):
        e = effects(df[df["n_vars"] == n], mechanism).set_index("arch").loc[archs]
        lines += [rf"        \textit{{{n} variables}} & \multicolumn{{{len(archs)}}}{{c}}{{}} \\",
                  rf"        \quad $\Delta${label} (paired) & " + " & ".join(signed(v) for v in e["paired"]) + r" \\",
                  r"        \quad Converged (\%) & "
                  + " & ".join(arrow(a, b, percent) for a, b in zip(e["conv_off"], e["conv_on"])) + r" \\",
                  rf"        \quad Best {label} & "
                  + " & ".join(arrow(a, b, short) for a, b in zip(e["best_off"], e["best_on"])) + r" \\"]
    return lines


def table(df, split, mechanisms, label, intro):
    archs = [a for a in ARCHS.values() if a in set(df["arch"])] + ["All"]
    lines = [r"\begin{table}[ht]", r"    \centering", r"    \small", r"    \setlength{\tabcolsep}{3pt}",
             r"    \begin{tabular}{l" + "c" * (len(archs) - 1) + "|c}", r"    \toprule",
             "        ~ & " + " & ".join(rf"\textbf{{{a}}}" for a in archs) + r" \\"]
    for m in mechanisms:
        if len(mechanisms) > 1:
            lines += [r"    \midrule", rf"        \multicolumn{{{len(archs) + 1}}}{{c}}{{\textit{{{MECHANISMS[m]}}}}} \\"]
        lines += [r"    \midrule"] + block(df, m, archs)
    name, threshold, higher = METRICS[METRIC]
    caption = (intro + r" in the \emph{Single} grid search, on " + SPLIT_TEXT[split] + r", over the "
               rf"{WORDS[df[df['n_vars'] == 3]['dataset'].nunique()]} three-variable (model size \emph{{small}}) and "
               rf"{WORDS[df[df['n_vars'] == 5]['dataset'].nunique()]} five-variable datasets (\emph{{medium}}) and all "
               rf"other hyperparameters. {name} is computed per graph and averaged; $\Delta${name} (paired) is the "
               r"mean difference between configurations "
               r"that differ only in the mechanism (on $-$ off). Converged is the share of configurations with "
               rf"{'an' if name[0] in 'AaNn' else 'a'} {name} of {'at least' if higher else 'at most'} {threshold}, and "
               rf"Best {name} the mean over datasets of the best "
               r"configuration, each without $\to$ with the mechanism. Both passes of the grid use the same "
               r"seed and are averaged. CM uses the corrected output head (\cref{cp:architectures}) and was run once.")
    lines += [r"    \bottomrule", r"    \end{tabular}", rf"    \caption{{{caption}}}", rf"    \label{{{label}}}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "ts1", "ts2"], default="val")
    parser.add_argument("--metric", choices=list(METRICS), default="nauprc")
    args = parser.parse_args()
    global METRIC
    METRIC = args.metric

    df = configurations(args.split)
    print(f"{len(df)} configurations, {df['dataset'].nunique()} datasets")
    summary = []
    for m in MECHANISMS:
        for n in (3, 5):
            sub = df[df["n_vars"] == n]
            summary.append(effects(sub, m).assign(mechanism=m, n_vars=n, by="arch"))
            if m != "corr_input":
                for ci in (True, False):
                    summary.append(effects(sub[sub["corr_input"] == ci], m)
                                   .assign(mechanism=m, n_vars=n, by=f"arch|CI={ci}"))
        summary.append(effects(df, m, "dataset").assign(mechanism=m, by="dataset"))
    summary = pd.concat(summary)
    msuffix = "" if args.metric == "nauprc" else f"_{args.metric}"
    summary.to_csv(THESIS / "results" / f"helpers_{args.split}{msuffix}.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))

    suffix = ("" if args.split == "val" else f"_{args.split}") + msuffix
    outputs = {f"tab_cp_ci{suffix}.tex": table(df, args.split, ["corr_input"], "tab:cp:ci" + suffix.replace("_", ":"),
                                               "Effect of correlation injection"),
               f"tab_cp_helpers{suffix}.tex": table(df, args.split, ["regression_head", "model.corr_regularization"],
                                                    "tab:cp:helpers" + suffix.replace("_", ":"),
                                                    "Effect of the edge-count head and of correlation regularization")}
    for name, out in outputs.items():
        (THESIS / "outputs" / name).write_text(out)
        print(out)


if __name__ == "__main__":
    main()
