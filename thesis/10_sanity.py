"""Score the sanity-check runs (Ch. 7.1) and write the result table.

    # 1. data (once):  python thesis/make_shuffled.py
    # 2. train:        python train.py +experiment=thesis_sanity_<task> hparams_search=thesis_sanity
    #                  for task in graphs, nonlinear, coefficients, shuffled
    # 3. score:        python thesis/10_sanity.py

Every run is scored in eval mode on the test split of its dataset:
graphs, shuffled  AUROC of the predicted window graph (shuffled: trained on permuted labels,
                  tested on true labels, so chance level is the expected result)
nonlinear         AUROC of the linear-vs-nonlinear prediction (distinguish_mode)
coefficients      MSE / MAE of the predicted coefficient tensor; the reference is predicting the
                  mean training coefficients for every sample

Writes thesis/results/sanity.csv (one row per run) and thesis/outputs/tab_cp_sanity.tex.
"""
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ARCHS, PATHS, THESIS, auroc, best_checkpoint, deterministic_ds, load_model, run_config

TASKS = ["graphs", "nonlinear", "coefficients", "shuffled"]
SANITY_ROOT = PATHS["new_runs_root"]


@torch.no_grad()
def predict_raw(model, cfg):
    """Raw network outputs and labels on the test split, in eval mode."""
    model.eval()
    ds = deterministic_ds(
        ds_path=cfg["data"]["test_ds_name"],
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


def mean_coefficient_reference(cfg):
    train = pickle.load(open(cfg["data"]["train_ds_name"], "rb"))
    test = pickle.load(open(cfg["data"]["test_ds_name"], "rb"))
    mean = torch.stack([s[1] for s in train]).float().mean(0)
    y = torch.stack([s[1] for s in test]).float()
    return ((y - mean) ** 2).mean().item(), (y - mean).abs().mean().item()


def score_task(task):
    rows = []
    for hydra_dir in sorted((SANITY_ROOT / f"sanity_{task}").rglob(".hydra")):
        run_dir = hydra_dir.parent
        ckpt = best_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] no checkpoint: {run_dir}")
            continue
        cfg = run_config(run_dir)
        out, lab = predict_raw(load_model(ckpt), cfg)
        row = {"task": task, "arch": ARCHS[cfg["model"]["model_type"]], "seed": cfg["seed"],
               "corr_input": cfg["corr_input"], "run_dir": str(run_dir.relative_to(SANITY_ROOT))}
        if task == "coefficients":
            row["mse"] = ((out - lab) ** 2).mean().item()
            row["mae"] = (out - lab).abs().mean().item()
            row["ref_mse"], row["ref_mae"] = mean_coefficient_reference(cfg)
        else:
            row["auroc"] = auroc(torch.sigmoid(out).flatten(), lab.flatten())
        rows.append(row)
    return rows


def latex_table(df):
    """Rows: task x correlation injection; columns: architectures; mean (std) over seeds."""
    metric = {"graphs": "auroc", "nonlinear": "auroc", "shuffled": "auroc", "coefficients": "mse"}
    label = {"graphs": "Graphs (AUROC)", "nonlinear": "Nonlinearity (AUROC)",
             "coefficients": "Coefficients (MSE)", "shuffled": "Shuffled labels (AUROC)"}
    archs = [a for a in ARCHS.values() if a in set(df["arch"])]
    lines = [r"\begin{tabular}{ll" + "c" * len(archs) + "}", r"\toprule",
             "Task & CI & " + " & ".join(archs) + r" \\", r"\midrule"]
    for task in TASKS:
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        for ci in (True, False):
            cells = []
            for a in archs:
                v = sub[(sub["arch"] == a) & (sub["corr_input"] == ci)][metric[task]]
                cells.append(f"{v.mean():.3f} ({v.std():.3f})" if len(v) else "--")
            lines.append(f"{label[task] if ci else ''} & {'yes' if ci else 'no'} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    df = pd.DataFrame([r for task in TASKS for r in score_task(task)])
    if df.empty:
        raise SystemExit(f"no sanity runs found below {SANITY_ROOT}")
    (THESIS / "results").mkdir(exist_ok=True)
    (THESIS / "outputs").mkdir(exist_ok=True)
    df.to_csv(THESIS / "results" / "sanity.csv", index=False)
    (THESIS / "outputs" / "tab_cp_sanity.tex").write_text(latex_table(df))
    metrics = [c for c in ("auroc", "mse", "mae", "ref_mse") if c in df]
    print(df.groupby(["task", "corr_input", "arch"])[metrics].agg(["mean", "std", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
