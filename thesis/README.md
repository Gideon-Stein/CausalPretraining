# Thesis reproduction (CP v1, Ch. 7.1)

This folder reproduces every CP v1 number in the dissertation *On Causal Discovery for
Real-World Applications*. It only adds code: the released model, data and training code
is used as is, so all original checkpoints load and score identically.

## What lives where

| What | Where | In git |
|---|---|---|
| Code, configs, small result CSVs | this repo (`experimental_results/`, `thesis/`) | yes |
| Original datasets + baseline scores | `paths.yaml:store_root` (read-only) | checksums in `manifest/*.md5` |
| Training runs + checkpoints | `paths.yaml:runs_root` (`cp_1`) | mapping in `manifest/runs.yaml` |

`data/generate_synthetic_ds.py` is unseeded, so regenerated datasets differ from the
originals. Always use the store.

## Naming

The result files use the original dataset names and swapped test-set columns.
`common.py` translates both; scripts should never use the raw names directly.

| Thesis | Folder | | Thesis | Stored column |
|---|---|---|---|---|
| SL | `base` | | Test-Set 1 (in-distribution, `<ds>_additional/simple_test.p`) | `Test2_AUROC` |
| ML | `big_base_mirror` | | Test-Set 2 (shifted, `<ds>/simple_test.p`) | `Test1_Auroc` |
| SNL | `poly` | | | |
| MNL | `big_poly` | | | |
| LNL | `nl` | | | |
| XLNL | `big_nl` | | | |

Scaling: 5 vars = `big_base_mirror`, 7 = `scale_up_2`, 10 = `scale_up`, 12 = `scale_up_4`, 15 = `scale_up_3`.

## Scripts

| Script | Produces |
|---|---|
| `00_verify.py` | checks data checksums, run mapping, architectures, and re-scores sample runs |
| `ct_baseline.py` | `results/ct_baseline.csv`: CT reference recomputed on the store data, next to the stored paper values |
| `rescore.py --mode eval\|legacy` | `results/rescored_<mode>.csv`: every std run, incl. runs never scored |
| `make_shuffled.py` | shuffled-label control set in `paths.yaml:derived_root` (checksums: `manifest/derived.md5`) |
| `10_sanity.py` | `results/sanity.csv`, `outputs/tab_cp_sanity.tex` |

Run everything from the repo root in the `causal_pretraining` environment, e.g.
`python thesis/00_verify.py`.

## Scoring: legacy vs eval

The original scoring (`slurm/slurm_calc_final_scoring.py`) never called `model.eval()`:
BatchNorm used test-batch statistics and dropout stayed active, on shuffled batches. All
stored CSVs (and the paper) therefore depend on the scoring batch size, which also differed
between scoring sessions (`legacy_bs` in `manifest/runs.yaml`). `common.predict(legacy=True)`
reproduces this; the default is proper eval-mode inference. The MLP is affected most.

## Sanity checks (new runs)

Same setup as the original preliminary studies (small models, batch 64, lr 1e-4, wd 0.01,
500 epochs, no early stopping), now for all 5 architectures x 5 seeds x with/without
correlation injection:

```bash
python thesis/make_shuffled.py            # once
for t in graphs nonlinear coefficients shuffled; do
  python train.py +experiment=thesis_sanity_$t hparams_search=thesis_sanity
done
python thesis/10_sanity.py
```

Configs: `config/experiment/thesis_sanity_*.yaml`, sweep: `config/hparams_search/thesis_sanity.yaml`.
Runs land in `paths.yaml:new_runs_root`. The grid runs of the original experiments were
launched with `hparams_search` files that were later edited in place (`joint_slurm.yaml`,
`slurm_exp2.yaml`, `kuramoto.yaml`); the actual grids are recorded in each run's
`.hydra/overrides.yaml` in `cp_1`.
