"""Score the sanity-check runs (Ch. 7.1) and write the result tables.

    # 1. data (once):  python thesis/make_shuffled.py
    # 2. train:        python train.py +experiment=thesis_sanity_<task> hparams_search=thesis_sanity
    #                  for task in graphs, nonlinear, coefficients, coefficients_wmse, shuffled
    # 2b. CM with the corrected head (convM_head_activation, see model/conv.py), same sweeps:
    #                  python train.py +experiment=thesis_sanity_<task> hparams_search=thesis_sanity \
    #                    model.model_type=convM model.convM_head_activation=True \
    #                    'hydra.sweep.dir=<new_runs_root>/cm_fixed/sanity_<task>/${now:...}'
    # 3. score:        python thesis/10_sanity.py [--tasks graphs ...] [--cm-fixed-only]

The sanity checks train for a fixed 500 epochs without early stopping, so every run is scored
with its last checkpoint, in eval mode, on three splits:
train             the training split itself (is the task fitted at all, or only memorized?)
in distribution   coefficients from the training range: the validation split of the sanity
                  datasets (no longer used for checkpoint selection), and SL Test-Set 1
                  (base_additional) for the shuffled control, whose validation labels are permuted
shifted           the test split, with coefficients from [0.2, 0.3] instead of [0.3, 0.5]
                  (shuffled: SL Test-Set 2, the original test set with true labels)

graphs            nSHD and nAUPRC per graph (common.metrics, thesis ch. 2.7) and AUROC (pooled) of the edge
                  probabilities, plus graph identification: each sample is
                  assigned the nearest of the fixed graphs of the dataset (Euclidean distance
                  between edge probabilities and support), reported as accuracy and as macro
                  accuracy (mean over graphs; the graph frequencies are unbalanced).
                  References: an input-blind predictor that outputs the per-entry training
                  frequency of every edge (nSHD, nAUPRC, AUROC) and always predicts the most frequent training
                  graph (accuracy); any constant prediction has macro accuracy 1/#graphs.
shuffled          nSHD, nAUPRC, AUROC (trained on permuted labels, tested on true labels; chance is expected).
                  Reference: the input-blind predictor above, fitted on the permuted labels.
nonlinear         AUROC of the linear-vs-nonlinear prediction (distinguish_mode); reference 0.5.
coefficients(_wmse)  R^2 = 1 - SSE / SST per split on the non-zero entries of the fixed graph, with SST
                  around the per-entry mean of that split (R^2 = 0: a constant at the split mean;
                  R^2 = 1: exact recovery). Every series is min-max scaled per variable after
                  simulation (data/generator.py), which rescales a cross-link j -> i by
                  range_j / range_i, so only the self-links are identifiable from the input.
                  Reported: r2 (self-links, the table metric), r2_cross, r2_all, mse (all non-zero).
                  Reference: per-series least squares (with intercept) on the known support, an
                  oracle that is given the graph (ref_r2, on the self-links).

CM runs are read from both sanity_<task> (original linear head, head_activation False) and
cm_fixed/sanity_<task> (head with ReLU, head_activation True). Both are kept in the CSVs; the
tables report CM with the corrected head only. --cm-fixed-only scores just the cm_fixed runs and
replaces the corresponding rows of the stored CSVs.

Writes thesis/results/sanity_<task>.csv (one row per run and split),
thesis/outputs/tab_cp_sanity_graphs.tex (tab:cp:sanity-graphs) and
thesis/outputs/tab_cp_sanity.tex (tab:cp:sanity-further, all other tasks).
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (ARCHS, PATHS, THESIS, auroc, deterministic_ds, load_model, metrics, number, run_config,
                    test_set_path)

TASKS = ["graphs", "nonlinear", "coefficients", "coefficients_wmse", "shuffled"]
SPLITS = {"train": "Training", "in": "In distribution", "shifted": "Shifted"}
SHORT = {"train": "Train", "in": "In-dist.", "shifted": "Shifted"}  # split labels in the tables
GRAPH_TABLE_SPLITS = ["in", "shifted"]  # the training split of the graph task is reported in the text only
SANITY_ROOT = PATHS["new_runs_root"]
CM_FIXED = "cm_fixed"  # subfolder of SANITY_ROOT with the CM runs that use the corrected head
KEYS = ["task", "arch", "head_activation", "seed", "corr_input", "split"]


def split_paths(task, cfg):
    train, shifted = cfg["data"]["train_ds_name"], cfg["data"]["test_ds_name"]
    if task == "shuffled":  # training labels are the permuted ones the model was trained on
        return {"train": train, "in": str(test_set_path("SL", 1)), "shifted": shifted}
    return {"train": train, "in": cfg["data"]["val_ds_name"], "shifted": shifted}


def last_checkpoint(run_dir):
    return next(Path(run_dir).rglob("last.ckpt"), None)


@torch.no_grad()
def predict_raw(model, cfg, path):
    """Raw network outputs and labels on one split, in eval mode."""
    model.eval()
    ds = deterministic_ds(
        ds_path=path,
        binary=cfg["data"]["binary"],
        regression=False,
        corr_input=cfg["corr_input"],
        distinguish_mode=cfg["distinguish_mode"],
    )
    outs, labs = [], []
    for X, Y in DataLoader(ds, batch_size=50, shuffle=False):
        out = model.model(X)
        if cfg["distinguish_mode"]:
            out = out[1]  # distinguish logit lives in the regression head
        outs.append(out)
        labs.append(Y)
    return torch.cat(outs), torch.cat(labs)


# ---------------------------------------------------------------------------
# References and metrics
# ---------------------------------------------------------------------------

def coefficients(path):
    return torch.stack([s[1] for s in pickle.load(open(path, "rb"))]).float()


def support(path):
    """Binary edge labels of all samples of a split, shaped (samples, edges)."""
    return (coefficients(path).abs() > 0).float().flatten(1)


def graph_index(lab, graphs):
    """Index of every sample's graph in `graphs`; fails if a sample has an unknown graph."""
    match = (lab[:, None] == graphs[None]).all(2)
    if not match.any(1).all():
        raise ValueError("split contains graphs that do not occur in training")
    return match.float().argmax(1)


def identification(pred, true, n_graphs):
    acc = (pred == true).float().mean().item()
    macro = sum((pred[true == g] == g).float().mean().item() for g in range(n_graphs)) / n_graphs
    return acc, macro


def graph_scores(prob, lab):
    """nSHD, nAUPRC (per graph) and pooled AUROC of edge probabilities shaped (samples, ...)."""
    m = metrics(prob.flatten(1), lab.flatten(1))
    return {k: m[k] for k in ("nshd", "nauprc", "auroc")}


def blind(train, lab):
    """Scores of an input-blind predictor that outputs the per-entry training frequency of every edge."""
    return graph_scores(train.mean(0).expand_as(lab), lab)


class GraphSet:
    """The fixed graphs of a dataset (from its training split) and the references on one split."""

    def __init__(self, train_path, path):
        train, lab = support(train_path), support(path)
        self.graphs = torch.unique(train, dim=0)
        self.true = graph_index(lab, self.graphs)
        majority = graph_index(train, self.graphs).mode().values
        acc, macro = identification(torch.full_like(self.true, majority), self.true, len(self.graphs))
        self.reference = {**blind(train, lab), "acc": acc, "macro": macro}

    def score(self, prob):
        pred = torch.cdist(prob.flatten(1), self.graphs).argmin(1)
        return identification(pred, self.true, len(self.graphs))


def least_squares(x, support):
    """Per-equation OLS (with intercept) of one series x (T, V) on the known support.

    Labels are links[effect, cause, l], where index l is lag L - l (data/generator.py:links_to_ts).
    """
    V, _, L = support.shape
    T = x.shape[0]
    est = torch.zeros(support.shape)
    for i in range(V):
        cols = [(j, l) for j in range(V) for l in range(L) if support[i, j, l]]
        if cols:
            A = torch.stack([x[l: T - L + l, j] for j, l in cols] + [torch.ones(T - L)], 1)
            coef = torch.linalg.lstsq(A, x[L:, i:i + 1]).solution[:-1, 0]
            for (j, l), c in zip(cols, coef):
                est[i, j, l] = c
    return est


class CoefficientReference:
    """Masks of the fixed graph (self- and cross-links) and the least-squares reference on one split."""

    def __init__(self, train_path, path):
        self.support = coefficients(train_path).mean(0) != 0
        diagonal = torch.eye(self.support.shape[0], dtype=torch.bool)[:, :, None].expand_as(self.support)
        self.masks = {"r2": self.support & diagonal, "r2_cross": self.support & ~diagonal, "r2_all": self.support}
        self.lab = coefficients(path)
        series = [torch.as_tensor(s[0]).float() for s in pickle.load(open(path, "rb"))]
        self.reference = self.r2(torch.stack([least_squares(x, self.support) for x in series]), self.lab)

    def r2(self, out, lab):
        res = {}
        for name, mask in self.masks.items():
            o, l = out[:, mask], lab[:, mask]
            res[name] = 1 - ((o - l) ** 2).sum().item() / ((l - l.mean(0)) ** 2).sum().item()
        return res

    def score(self, out, lab):
        assert torch.allclose(lab, self.lab), "dataset labels differ from the stored coefficients"
        return {**self.r2(out, lab), "mse": ((out - lab)[:, self.support] ** 2).mean().item(),
                **{f"ref_{k}": v for k, v in self.reference.items()}}


class References:
    """Per-split references, computed once per task and split."""

    def __init__(self, task):
        self.task, self.cache = task, {}

    def __call__(self, train_path, path):
        key = (train_path, path)
        if key not in self.cache:
            if self.task == "graphs":
                self.cache[key] = GraphSet(train_path, path)
            elif self.task.startswith("coefficients"):
                self.cache[key] = CoefficientReference(train_path, path)
            elif self.task == "shuffled":
                self.cache[key] = blind(support(train_path), support(path))
            else:
                self.cache[key] = {"auroc": 0.5}  # balanced binary label, any constant prediction
        return self.cache[key]


def score_split(task, out, lab, ref):
    if task.startswith("coefficients"):
        return ref.score(out, lab)
    prob = torch.sigmoid(out)
    if task == "nonlinear":  # one binary label per sample, not a graph
        return {"auroc": auroc(prob.flatten(), lab.flatten()), "ref_auroc": ref["auroc"]}
    row = graph_scores(prob, lab)
    if task == "graphs":
        row["acc"], row["macro"] = ref.score(prob)
        row.update({f"ref_{k}": v for k, v in ref.reference.items()})
    else:
        row.update({f"ref_{k}": v for k, v in ref.items()})
    return row


def run_folders(task, cm_fixed_only=False):
    folders = [SANITY_ROOT / CM_FIXED / f"sanity_{task}"]
    if not cm_fixed_only:
        folders.insert(0, SANITY_ROOT / f"sanity_{task}")
    return folders


def score_task(task, cm_fixed_only=False):
    rows, refs = [], References(task)
    hydra_dirs = [h for f in run_folders(task, cm_fixed_only) for h in sorted(f.rglob(".hydra"))]
    for hydra_dir in hydra_dirs:
        run_dir = hydra_dir.parent
        ckpt = last_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] no checkpoint: {run_dir}")
            continue
        cfg = run_config(run_dir)
        model = load_model(ckpt)
        run = {"task": task, "arch": ARCHS[cfg["model"]["model_type"]],
               "head_activation": cfg["model"].get("convM_head_activation", False), "seed": cfg["seed"],
               "corr_input": cfg["corr_input"], "epoch": torch.load(ckpt, map_location="cpu")["epoch"],
               "run_dir": str(run_dir.relative_to(SANITY_ROOT))}
        for split, path in split_paths(task, cfg).items():
            out, lab = predict_raw(model, cfg, path)
            rows.append({**run, "split": split, **score_split(task, out, lab, refs(cfg["data"]["train_ds_name"], path))})
        print(f"[{task}] {run['run_dir']} {run['arch']} seed={run['seed']} ci={run['corr_input']}", flush=True)
    df = pd.DataFrame(rows)
    if not df.empty and df.duplicated(KEYS).any():
        raise SystemExit(f"sanity_{task} contains the same run twice (several sweeps?):\n"
                         f"{df[df.duplicated(KEYS, keep=False)].sort_values(KEYS).to_string()}")
    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def reported(df):
    """Rows that enter the tables: CM only with the corrected head."""
    return df[(df["arch"] != ARCHS["convM"]) | df["head_activation"]]


def fmt(x):
    return number(x)  # handles the sign ($-$) and -.000


def cell(v):
    if not len(v):
        return "--"
    return fmt(v.mean()) if len(v) == 1 else f"{fmt(v.mean())} ({number(v.std())})"


def seeds_label(df):
    n = df.groupby(["task", "arch", "corr_input", "split"]).size()
    return str(n.iloc[0]) if n.nunique() == 1 else f"{n.min()}--{n.max()}"


def header(archs, first):
    return [r"\begin{table}[ht]", r"    \centering", r"    \small", r"    \setlength{\tabcolsep}{3.5pt}",
            r"    \begin{tabular}{" + first + "c" * len(archs) + "|c}",
            r"    \toprule",
            "        " + " & ".join(["~"] * (len(first) - 1) + ["CI"] + [rf"\textbf{{{a}}}" for a in archs])
            + r" & \textbf{Ref.} \\",
            r"    \midrule"]


def footer(caption, label):
    return [r"    \bottomrule", r"    \end{tabular}", rf"    \caption{{{caption}}}", rf"    \label{{{label}}}",
            r"\end{table}"]


def block(df, archs, metric, ref, labels):
    """Two rows (with / without correlation injection); `labels` are the multirow cells in front."""
    lines = []
    for ci in (True, False):
        sub = df[df["corr_input"] == ci]
        cells = " & ".join(cell(sub.loc[sub["arch"] == a, metric]) for a in archs)
        front = " & ".join(labels if ci else ["~"] * len(labels))
        ref_cell = rf"\multirow{{2}}{{*}}{{{ref}}}" if ci else "~"
        lines.append(f"        {front} & {'yes' if ci else 'no'} & {cells} & {ref_cell} \\\\")
    return lines


def graphs_table(df):
    """Rows: split x metric x correlation injection; columns: architectures and the input-blind reference."""
    archs = [a for a in ARCHS.values() if a in set(df["arch"])]
    metrics = {"nauprc": "nAUPRC", "nshd": "nSHD", "macro": "Macro acc."}
    n_graphs = round(1 / df["ref_macro"].iloc[0])
    caption = (rf"Sanity check \emph{{Distinguishing between graphs}}: mean over {seeds_label(df)} seeds, "
               r"with the standard deviation in brackets, with (\emph{yes}) and without (\emph{no}) "
               r"correlation injection (CI). The in-distribution split (\emph{In-dist.}) draws the coefficients "
               r"from the training range $[0.3,0.5]$, the shifted split from $[0.2,0.3]$. Macro accuracy assigns every sample the nearest of the "
               rf"{n_graphs} graphs and averages the accuracy over the graphs, which occur with unequal "
               r"frequency. nSHD (lower is better) and nAUPRC are computed per graph and averaged. "
               r"\textbf{Ref.} is an input-blind predictor that outputs the training frequency of "
               rf"every edge; any constant prediction has a macro accuracy of 1/{n_graphs}.")
    lines = header(archs, "lll")
    for i, split in enumerate(GRAPH_TABLE_SPLITS):
        sub, split_label = df[df["split"] == split], SHORT[split]
        for j, (metric, label) in enumerate(metrics.items()):
            first = rf"\multirow{{{2 * len(metrics)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{split_label}}}}}}}" if j == 0 else "~"
            lines += block(sub, archs, metric, number(sub[f"ref_{metric}"].iloc[0]),
                           [first, rf"\multirow{{2}}{{*}}{{{label}}}"])
            if j < len(metrics) - 1:
                lines.append(rf"    \cmidrule{{2-{len(archs) + 4}}}")
        if i < len(GRAPH_TABLE_SPLITS) - 1:
            lines.append(r"    \midrule")
    return "\n".join(lines + footer(caption, "tab:cp:sanity-graphs")) + "\n"


OTHER = {"nonlinear": ("Nonlinearity", "AUROC", "auroc"), "shuffled": ("Shuffled labels", "nAUPRC", "nauprc"),
         "coefficients": ("Coefficients (MSE)", "$R^2$ of the self-links", "r2"),
         "coefficients_wmse": ("Coefficients (WMSE)", "$R^2$ of the self-links", "r2")}


def other_table(df):
    """Rows: task x split x correlation injection; columns: architectures and the reference."""
    archs = [a for a in ARCHS.values() if a in set(df["arch"])]
    tasks = [t for t in OTHER if t in set(df["task"])]
    caption = (rf"Further sanity checks: mean over {seeds_label(df)} seeds, with the standard deviation in "
               r"brackets, with (\emph{yes}) and without (\emph{no}) correlation injection (CI), on the "
               r"training split (\emph{Train}), the in-distribution split (\emph{In-dist.}, coefficients from the "
               r"training range $[0.3,0.5]$), and the shifted split (coefficients from $[0.2,0.3]$); for the shuffled "
               r"labels, these are \textit{Test-Set 1} and \textit{Test-Set 2} of \textbf{SL}, and the training "
               r"split carries the permuted labels. For the coefficients, $R^2$ is computed "
               r"per split on the self-links of the fixed graph, relative to their mean on that split ($R^2=0$); the shifted split is omitted there (see text). "
               r"\textbf{Ref.} is an input-blind predictor for the AUROC and the nAUPRC and, for the coefficients, a "
               r"least-squares fit on the known graph.")
    n_cols = len(archs) + 3
    lines = header(archs, "ll")
    for i, task in enumerate(tasks):
        name, metric_label, metric = OTHER[task]
        lines += [rf"        \multicolumn{{{n_cols}}}{{c}}{{\textit{{{name}}}, {metric_label}}} \\", r"    \midrule"]
        # the shifted coefficients span only [0.2, 0.3]: R^2 around their mean is dominated by the
        # offset from the training range (even least squares is below 0), so it is reported in the text
        splits = [s for s in SPLITS if not (metric == "r2" and s == "shifted")]
        for j, split in enumerate(splits):
            sub = df[(df["task"] == task) & (df["split"] == split)]
            ref = fmt(sub[f"ref_{metric}"].iloc[0])
            lines += block(sub, archs, metric, ref, [rf"\multirow{{2}}{{*}}{{{SHORT[split]}}}"])
            if j < len(splits) - 1:
                lines.append(rf"    \cmidrule{{2-{n_cols}}}")
        if i < len(tasks) - 1:
            lines.append(r"    \midrule")
    return "\n".join(lines + footer(caption, "tab:cp:sanity-further")) + "\n"


def summary(df, metrics):
    metrics = [m for m in metrics if m in df]
    return df.groupby(["task", "split", "corr_input", "arch"])[metrics].agg(["mean", "std", "count"]).round(3)


def load_csv(path):
    df = pd.read_csv(path)
    if "head_activation" not in df:  # CSVs written before the corrected CM head
        df.insert(2, "head_activation", False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS,
                        help="tasks to (re)score; the tables also use the stored CSVs of the other tasks")
    parser.add_argument("--cm-fixed-only", action="store_true",
                        help="score only the cm_fixed runs and merge them into the stored CSVs")
    args = parser.parse_args()

    results, outputs = THESIS / "results", THESIS / "outputs"
    results.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    for task in args.tasks:
        df = score_task(task, args.cm_fixed_only)
        if df.empty:
            print(f"[{task}] no runs below {run_folders(task, args.cm_fixed_only)}")
            continue
        if args.cm_fixed_only:
            old = load_csv(results / f"sanity_{task}.csv")
            old = old.drop(columns="run_dir").merge(df[KEYS], on=KEYS, how="left", indicator=True)
            keep = load_csv(results / f"sanity_{task}.csv")[(old["_merge"] == "left_only").values]
            df = pd.concat([keep, df], ignore_index=True)
        df.to_csv(results / f"sanity_{task}.csv", index=False)

    scored = {t: reported(load_csv(results / f"sanity_{t}.csv"))
              for t in TASKS if (results / f"sanity_{t}.csv").exists()}
    if "graphs" in scored:
        table = graphs_table(scored["graphs"])
        (outputs / "tab_cp_sanity_graphs.tex").write_text(table)
        print(table)
        print(summary(scored["graphs"], ["nauprc", "nshd", "auroc", "acc", "macro"]).to_string())
    others = [df for t, df in scored.items() if t != "graphs"]
    if others:
        df = pd.concat(others)
        table = other_table(df)
        (outputs / "tab_cp_sanity.tex").write_text(table)
        print(table)
        print(summary(df, ["nauprc", "nshd", "auroc", "r2", "mse"]).to_string())


if __name__ == "__main__":
    main()
