"""Build tab:cp:kuramoto (learning a different generative family, Ch. 7.1).

    python thesis/ct_baseline.py                # only if results/ct_baseline.csv is missing
    python thesis/rescore.py --mode eval --only kuramoto
    python thesis/40_kuramoto.py

Reads thesis/results/rescored_eval.csv and ct_baseline.csv. Prints the LaTeX table to stdout and
the number of runs per cell to stderr, and writes thesis/outputs/tab_cp_kuramoto.tex. Cells are
mean (sample std) over reruns of nAUPRC and nSHD (per graph, common.metrics), in eval mode and without
self-links. The mean AUROC (pooled, as reported by Lowe et al.) is printed for the text. Kuramoto has
only one test set, which the result files call Test-Set 2.
"""
import sys

import pandas as pd

from common import ARCHS, THESIS, number

SIZES = {"medium": "Medium", "big": "Big"}
METRICS = {"nauprc": "nAUPRC", "nshd": "nSHD"}
CAPTION = (r"Mean nAUPRC and nSHD (lower is better) on the Kuramoto test set, per graph and ignoring "
           r"self-links, with the standard deviation over reruns in brackets.")


def size_row(df, size, metric):
    cells = []
    for arch in ARCHS.values():
        v = df.loc[(df["size"] == size) & (df["arch"] == arch), f"ts2_{metric}"]
        cells.append(f"{number(v.mean())} ({number(v.std())})" if len(v) else "--")
    return f"        ~ & \\textbf{{{SIZES[size]}}} & " + " & ".join(cells) + r" \\"


def latex_table(df, ct):
    header = " & ".join(rf"\textbf{{{a}}}" for a in ARCHS.values())
    lines = [r"\begin{table}[ht]", r"    \centering", r"    \begin{tabular}{llccccc}", r"    \toprule",
             rf"        & & {header} \\"]
    for metric, label in METRICS.items():
        rows = [size_row(df, size, metric) for size in SIZES]
        rows.append(rf"        ~ & \textbf{{CT}} & \multicolumn{{5}}{{c}}{{{number(ct[metric])}}} \\")
        rows[0] = rows[0].replace("        ~ &", rf"        \multirow{{3}}{{*}}{{{label}}} &", 1)
        lines += [r"    \midrule"] + rows
    lines += [r"    \bottomrule", r"    \end{tabular}", rf"    \caption{{{CAPTION}}}",
              r"\label{tab:cp:kuramoto}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    df = pd.read_csv(THESIS / "results" / "rescored_eval.csv")
    df = df[df["experiment"] == "kuramoto"]
    if df.empty:
        sys.exit("No Kuramoto runs yet: python thesis/rescore.py --mode eval --only kuramoto")

    ct = pd.read_csv(THESIS / "results" / "ct_baseline.csv")
    ct = ct[(ct["experiment"] == "kuramoto") & (ct["split"] == "ts2")].iloc[0]

    table = latex_table(df, ct)
    (THESIS / "outputs").mkdir(exist_ok=True)
    (THESIS / "outputs" / "tab_cp_kuramoto.tex").write_text(table)
    print(table)

    print("mean AUROC (pooled, for the comparison with Lowe et al.):", file=sys.stderr)
    print(df.groupby(["size", "arch"])["ts2_auroc"].agg(["mean", "std"]).round(3).to_string(), file=sys.stderr)
    print(f"CT AUROC {ct['auroc_scored']:.3f}", file=sys.stderr)
    counts = df.groupby(["size", "arch"]).size().unstack().reindex(index=list(SIZES), columns=list(ARCHS.values()))
    print(f"runs per cell:\n{counts.fillna(0).astype(int).to_string()}", file=sys.stderr)


if __name__ == "__main__":
    main()
