"""Shared helpers for reproducing the CP v1 numbers of the thesis.

Everything that translates between the names used in the code/result files and the
names used in the thesis lives here, so that no script has to know about it.
"""
import sys
from pathlib import Path

import pandas as pd
import torch
import torchmetrics
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
THESIS = REPO / "thesis"
sys.path.insert(0, str(REPO))

from data.generator import deterministic_ds  # noqa: E402
from model.model_wrapper import Architecture_PL  # noqa: E402

PATHS = {k: Path(v) for k, v in yaml.safe_load(open(THESIS / "paths.yaml")).items()}


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

# thesis dataset name -> original dataset folder
DATASETS = {
    "SL": "base",
    "ML": "big_base_mirror",
    "SNL": "poly",
    "MNL": "big_poly",
    "LNL": "nl",
    "XLNL": "big_nl",
    "Joint": "joint",
    "Kuramoto": "kuramoto",
}
SINGLE = ["SL", "ML", "SNL", "MNL", "LNL", "XLNL"]

# scaling experiment: number of variables -> original dataset folder
SCALE_UP = {5: "big_base_mirror", 7: "scale_up_2", 10: "scale_up", 12: "scale_up_4", 15: "scale_up_3"}

# preliminary-study datasets (sanity checks)
SANITY = {
    "graphs": "classify_graphs",
    "distinguish": "distinguish",
    "coefficients": "uncover_params",
    "coefficients_small": "uncover_params_small",
}

# model_type in configs -> thesis abbreviation
ARCHS = {
    "mlp": "MLP",
    "unidirectional": "uGRU",
    "bidirectional": "bGRU",
    "convM": "CM",
    "transformer": "Trf",
}

# The stored result columns are swapped w.r.t. the thesis:
# "Test1" was scored on <ds>/simple_test.p (shifted coefficients/noise) = thesis Test-Set 2,
# "Test2" on <ds>_additional/simple_test.p (in-distribution)            = thesis Test-Set 1.
RESULT_COLUMNS = {
    "Test2_AUROC": "ts1_auroc",
    "Test2_MSE": "ts1_mse",
    "Test1_Auroc": "ts2_auroc",
    "Test1_MSE": "ts2_mse",
    "Val_Auroc": "val_auroc",
    "Val_MSE": "val_mse",
    "Train_Auroc": "train_auroc",
    "Train_MSE": "train_mse",
}


def ds_folder(name):
    """Accept a thesis name (SL, Joint, ...) or an original folder name."""
    return DATASETS.get(name, name)


def test_set_path(name, test_set):
    """Path of thesis Test-Set 1 (in-distribution) or 2 (shifted) of a dataset."""
    folder = ds_folder(name)
    if test_set == 1:
        folder += "_additional"
    elif test_set != 2:
        raise ValueError(test_set)
    return PATHS["data_root"] / folder / "simple_test.p"


def load_results(csv):
    """Load a result summary CSV (one column per run) as a tidy frame with thesis names."""
    df = pd.read_csv(csv, index_col=0).T.rename(columns=RESULT_COLUMNS)
    df["arch"] = df["model.model_type"].map(ARCHS)
    for c in RESULT_COLUMNS.values():
        if c in df:
            df[c] = pd.to_numeric(df[c])
    if "seed" in df:
        df["seed"] = pd.to_numeric(df["seed"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Runs and scoring
# ---------------------------------------------------------------------------

def run_config(run_dir):
    return yaml.safe_load(open(Path(run_dir) / ".hydra" / "config.yaml"))


def best_checkpoint(run_dir):
    """The best (not the last) checkpoint of a run, as selected by the original scoring."""
    ckpts = [p for p in Path(run_dir).rglob("*.ckpt") if "step" in p.name]
    return ckpts[0] if ckpts else None


def load_model(ckpt):
    return Architecture_PL.load_from_checkpoint(ckpt, map_location="cpu").eval()


def legacy_batch_size(test_set):
    """Batch size the original scoring used: 500 for Test-Set 1 (via load_deterministic_ds),
    50 for Test-Set 2 (via the run's data module with batch_size overridden to 50)."""
    return 500 if test_set == 1 else 50


@torch.no_grad()
def predict(model, ds_path, binary=True, batch_size=50, legacy=False, seed=0):
    """Sigmoid edge probabilities and labels for a whole test file.

    Mirrors predict_batch in slurm/slurm_calc_final_scoring.py (classification output).

    legacy=True reproduces how the stored result CSVs were scored: the original script
    never called model.eval(), so BatchNorm used batch statistics and dropout was active,
    on shuffled batches. Scores then depend on batch size and order (use
    legacy_batch_size). legacy=False is proper inference in eval mode.
    """
    model.train(legacy)
    torch.manual_seed(seed)
    ds = deterministic_ds(
        ds_path=str(ds_path),
        binary=binary,
        regression=model.regression_head,
        corr_input=model.hparams.corr_input,
    )
    outs, labs = [], []
    for X, Y in DataLoader(ds, batch_size=batch_size, shuffle=legacy):
        out = model.model(X)
        if model.regression_head:
            out, Y = out[0], Y[0]
        outs.append(torch.sigmoid(out))
        labs.append(Y)
    return torch.cat(outs), torch.cat(labs)


def auroc(pred, lab):
    return torchmetrics.classification.BinaryAUROC()(pred, lab.long()).item()
