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
| `ct_baseline.py` | `results/ct_baseline.csv`: CT reference recomputed on the store data, next to the stored paper values (AUROC), plus nSHD/nAUPRC |
| `rescore.py --mode eval\|legacy` | `results/rescored_<mode>.csv`: every std run, incl. runs never scored (eval: nSHD, nAUPRC, AUROC per test set) |
| `make_shuffled.py` | shuffled-label control set in `paths.yaml:derived_root` (checksums: `manifest/derived.md5`) |
| `10_sanity.py [--tasks graphs ...]` | `results/sanity_<task>.csv`, `outputs/tab_cp_sanity_graphs.tex` (`tab:cp:sanity-graphs`), `outputs/tab_cp_sanity.tex` (other tasks) |
| `rescore_grid.py [--check N]` | `results/grid_eval.csv`: every *Single* grid run in eval mode on the validation split and both test sets (resumes; `--check` reproduces the stored legacy `Val_Auroc` on N runs) |
| `15_helpers.py [--split val\|ts1\|ts2] [--metric nauprc\|nshd\|auroc]` | `outputs/tab_cp_ci.tex` (`tab:cp:ci`, §7.1.3), `outputs/tab_cp_helpers.tex` (`tab:cp:helpers`, edge-count head and CR, App. A.3.1), `results/helpers_<split>.csv` (needs `rescore_grid.py`) |
| `20_breadth.py [--mode eval\|legacy]` | `outputs/tab_cp_synthetic.tex` (nAUPRC, `tab:cp:synthetic-nauprc`), `_nshd.tex` (`tab:cp:synthetic-nshd`, App.), `_auroc.tex` (not in the thesis), `tab_cp_joint_subsets.tex` (`tab:cp:joint-subsets`, Joint per dataset; cache `results/joint_by_subset*.csv`) (needs `ct_baseline.py`, `rescore.py`) |
| `30_scaling.py [--test-set 1\|2]` | `fig:cp:scaling` as `outputs/fig_cp_scaling.{pdf,png}`, `results/scaling_summary.csv`, `results/scaling_val.csv` (needs `rescore.py`) |
| `40_kuramoto.py` | prints `tab:cp:kuramoto`, `outputs/tab_cp_kuramoto.tex` (needs `ct_baseline.py`, `rescore.py`) |

Run everything from the repo root in the `causal_pretraining` environment, e.g.
`python thesis/00_verify.py`.

## Scoring: legacy vs eval

The original scoring (`slurm/slurm_calc_final_scoring.py`) never called `model.eval()`:
BatchNorm used test-batch statistics and dropout stayed active, on shuffled batches. All
stored CSVs (and the paper) therefore depend on the scoring batch size, which also differed
between scoring sessions (`legacy_bs` in `manifest/runs.yaml`). `common.predict(legacy=True)`
reproduces this; the default is proper eval-mode inference. The MLP is affected most.

## Metrics (thesis ch. 2.7)

The tables report **nSHD** (`scoring_tools.min_shd_exact`) and **nAUPRC** (`scoring_tools.auprc_normalized`),
computed **per graph** and averaged (`common.metrics`). Graphs without edges are excluded (both metrics are
undefined; SL has 25/33 of 500, Joint 104/110 of 3000). Kuramoto is scored without self-links. Joint
three-variable samples are scored at their true size: `data/generate_synthetic_ds.py --joint` padded them with
two N(0, 0.01) channels and zero labels at `[3:, :, :]`, `[:, 3:, :]` and lag index 0, so only `[:3, :3, 1:]`
is scored (`common.entry_mask`; found by the noise channels, 1500 per split, blocks 4-6 = SNL, LNL, SL; the
block order is ML, XLNL, MNL, SNL, LNL, SL, verified sample by sample against the Single datasets). The pooled
AUROC is kept in the CSVs for comparison with the paper; for Joint it is lower than the stored value, which
includes the trivially correct padding.

Eval-mode predictions of every scored run and split are cached in `paths.yaml:derived_root/predictions`
(`common.cached_predict`, keyed by run folder and checkpoint name), so metrics can be recomputed without the
models. Inference uses the GPU if available (agrees with CPU to about 1e-5).

All metrics come from TCD-Arena (`paths.yaml:tcd_scoring_tools`, loaded by file path in `common.py`
because TCD-Arena's `cd_zoo/tools/helpers.py` would shadow this repo's `helpers` package).
`common.auroc` (pooled over all graphs) is `scoring_tools.auroc`, i.e. TCD-Arena's "AUROC Joint". It
replaces the torchmetrics `BinaryAUROC` used before: the two agree to 1.5e-7 on 95 CP and CT predictions
(2026-11-11), and `00_verify.py` passes unchanged. The original scripts outside `thesis/` still use torchmetrics.

## Convergence helpers (existing grid)

The *Single* grid (`manifest/runs.yaml: grid.single_run_*`) varies all three helpers for every
architecture and dataset. Its stored scores (`synthetic_data_1_grid_results/run_*`) are validation
only and legacy-scored (`slurm/slurm_calc_grid_scoring.py`: no `model.eval()`, batch 16), so
`rescore_grid.py` rescores every run in eval mode. Legacy scoring is itself random (shuffled batches,
active dropout); `--check` reproduces the stored values to within about .02. Both grid passes use
seed 42 and are therefore averaged per configuration in `15_helpers.py`, not counted as reruns.
Model size goes with the dataset (3 variables: small, 5 variables: medium), so all effects are
reported separately for 3 and 5 variables. 111 of the 5760 grid runs have no checkpoint and are skipped.

## Scaling (new runs)

The original scaling grid (`grid.scaling` in `manifest/runs.yaml`) varies lr {1e-4, 1e-3} x batch
{16, 64, 256} x weight decay {0, 0.01} x CR {off, on} (CI on, edge-count head off, seed 42) per number of
variables, size and architecture. Gaps that are filled with new runs (in `paths.yaml:new_runs_root`):

| What | Config | Folder |
|---|---|---|
| XL-Trf, 10 vars: complete grid (only 7 of 24 configurations trained originally) | `+experiment=thesis_scaling_lcm_grid hparams_search=thesis_scaling_lcm_grid` | `scaling_lcm_grid/` |
| XL-Trf reruns (seeds 44-46 on 10 vars crashed originally; optionally 7 vars) | `+experiment=thesis_scaling_lcm hparams_search=thesis_scaling_lcm` | `scaling_lcm/` |
| Trf-deep reruns, 7 and 10 vars (missing from the original reruns) | `+experiment=thesis_scaling_deep hparams_search=thesis_scaling_lcm` | `scaling_deep/` |

The exact commands and the grid history are in the headers of the config files.

Status 2026-11-10: XL-Trf grid done (24/24, 9 diverged); its selection differs from the original
(lr 1e-3, bs 64, wd 0, CR **on**, eval AUROC about .72), so the 3 reruns in `scaling_lcm/` (old selection,
.63-.67) are superseded by 5 new reruns in `scaling_lcm_v2/` (command in the header of
`hparams_search/thesis_scaling_lcm_grid.yaml`). Trf-deep reruns not run yet. CM scaling: see below.

## Sanity checks (new runs)

Same setup as the original preliminary studies (small models, batch 64, lr 1e-4, wd 0.01,
500 epochs, no early stopping), now for all 5 architectures x 5 seeds x with/without
correlation injection:

```bash
python thesis/make_shuffled.py            # once
for t in graphs nonlinear coefficients coefficients_wmse shuffled; do
  python train.py +experiment=thesis_sanity_$t hparams_search=thesis_sanity   # Slurm, 50 jobs each
done
python thesis/10_sanity.py
```

Each sweep blocks until its jobs are done, so run them in separate `tmux` windows. A failed
run still shows COMPLETED in Slurm; check for `run/last.ckpt` in every run folder.
For a local run, drop `hparams_search` and fix the swept values, e.g.
`python train.py +experiment=thesis_sanity_graphs model.model_type=bidirectional seed=42 corr_input=True hydra.run.dir=<dir>`.

Scoring uses the last checkpoint (no early stopping by design) on three splits: the training
split (fitted at all, or only memorized?; only in `tab:cp:sanity-further` and the CSVs, the graph table
leaves it out), the validation split (coefficients from the training range, no longer used for selection) and the test split
(coefficients from [0.2, 0.3]); for the shuffled control, SL Test-Set 1 and Test-Set 2.
The coefficient tasks report R^2 on the self-links only: every series is min-max scaled per
variable (`data/generator.py`), so cross-link coefficients are not identifiable from the input.
Their reference is a least-squares fit on the known graph; the shifted split is left out of
the table because its coefficients span only [0.2, 0.3] (see the docstring).
`10_sanity.py` refuses to score a task folder that contains the same run twice, so remove
old or test sweeps from `sanity_<task>/` first.

## CM with corrected head (new runs)

The original CM head (`model/conv.py`) has no activation after the correlation injection, so its
edge scores are linear in the signed correlations and it cannot learn datasets with random
coefficient signs. `model.convM_head_activation=True` adds a ReLU after `fc1` (default `False` =
original head, needed to load the released weights). All CM numbers in 7.1 are retrained with it;
the runs go to `new_runs_root/cm_fixed/<experiment>/`, the old CM runs stay where they are.

| Experiment | Command | Folder | Status |
|---|---|---|---|
| Sanity checks (50 runs) | `python train.py +experiment=thesis_sanity_<task> hparams_search=thesis_sanity model.model_type=convM model.convM_head_activation=True 'hydra.sweep.dir=<new_runs_root>/cm_fixed/sanity_<task>/${now:%Y-%m-%d_%H-%M-%S}'` | `cm_fixed/sanity_<task>/` | done 2026-11-05, scored with `10_sanity.py --cm-fixed-only` |
| Single grid (6 x 96 runs) | `python train.py hparams_search=thesis_cm_single_grid data.ds_name=<ds> [model=medium.yaml n_vars=5 max_lags=3]` | `cm_fixed/single_grid/<ds>/` | done 2026-11-06, 576/576, scored with `rescore_grid.py --cm-fixed` -> `results/grid_eval_cm_fixed.csv` |
| Single reruns (6 x 10 seeds) | see header of `hparams_search/thesis_cm_single_std.yaml` | `cm_fixed/single_std/<ds>/` | done 2026-11-09, 60/60, `rescore.py --mode eval --only single_cm_fixed` |
| Joint grid (2 sizes x 24) | `python train.py hparams_search=thesis_cm_joint_grid` | `cm_fixed/joint_grid/2026-11-09_10-56-50` | done, 48/48 (the 5-run sweep from 2026-11-07 is an aborted duplicate, ignore) |
| Joint reruns (2 sizes x 10 seeds) | see header of `hparams_search/thesis_cm_joint_std.yaml` | `cm_fixed/joint_std/` | done 2026-11-11, 20/20 (big `2026-11-10_13-47-59`, medium `2026-11-10_13-48-09`; the two empty folders are aborted launches), `rescore.py --mode eval --only joint_cm_fixed`, table via `20_breadth.py` |
| Kuramoto grid (2 sizes x 24) | `python train.py hparams_search=thesis_cm_kuramoto_grid` | `cm_fixed/kuramoto_grid/` | open |
| Scaling grid (11 cells x 24) | see header of `hparams_search/thesis_cm_scaling_grid.yaml` (5 commands) | `cm_fixed/scaling_grid/<ds>/` | open |
| Scaling reruns (11 cells x 5 seeds) | after the grid | `cm_fixed/scaling_std/` | open; then `30_scaling.py` |
| Kuramoto reruns (2 sizes x 10 seeds) | see header of `hparams_search/thesis_cm_kuramoto_std.yaml` | `cm_fixed/kuramoto_std/` | after the grid; then `rescore.py --mode eval --only kuramoto_cm_fixed` and `40_kuramoto.py` |

## TODO: retrain with |r| correlation injection (v1 update)

Diagnosis (2026-11-11/12): the models get the signed `lagged_batch_corr`, while CT uses |r|. On the
random-sign datasets (ML, MNL, XLNL, all scaling sets) the networks have to learn an even function of r
per entry, which Single models mostly fail at (slow, seed-dependent plateau that early stopping cuts off).
Diagnostics on ML, TS2 nAUPRC (CT .993): head on correlations only, linear signed .03 vs |r| .993;
full models retrained, signed .455/.772/.004 (CM/Trf/bGRU) vs |r| .982/.987/.982. The 3-variable sets have
positive coefficients only, so nothing changes there.

Plan: change only the injection (|r| instead of r), keep the selected configs, early stopping and data.
Present it in 7.1 next to the original results and released weights as a v1 update, not as v2.

Code: flag `corr_abs` that applies `abs()` to `lagged_batch_corr` in `data/generator.py:deterministic_ds`
and, for rivers, in `causalrivers/experiments/causal_discovery_zoo` (`finetune_cp.py`,
`methods/causal_pretraining.py`). Configs from the selected runs with the flag set, plus manifest entries
so `rescore.py`, `20_breadth.py` and `30_scaling.py` pick them up.

| Priority | Runs | Seeds | Purpose | Status |
|---|---|---|---|---|
| 1 | Single ML, MNL, XLNL, selected medium configs (Trf, uGRU, bGRU, CM, MLP) | 5 | shows the fix (expected: CT level) | open |
| 2 | Joint big, all architectures | 5 | updated release next to the released weights | open |
| 3 | Scaling 7, 10, 15 vars, existing size configs | 3-5 | only synthetic regime where CT is not a ceiling (CT TS1: .892 at 10, .695 at 15 vars) | open |
| 4 | Rivers: fine-tune updated Joint big GRU on Bavaria Random-5 (select on Bavaria val), plus the same GRU from random init | 3 | real-data argument, does pretraining matter | open |

Skip Kuramoto (already above CT) and the 3-variable sets. The original rivers fine-tuning model was
selected on East Random-5 test F1, validation-selected runs give the same AUROC (see thesis memory
`cp-v1-rivers-finetune-check`), so 4 also fixes the selection.

`10_sanity.py` reads CM from both folders and keeps both in the CSVs (column `head_activation`);
the tables report only the corrected head.

Configs: `config/experiment/thesis_sanity_*.yaml`, sweep: `config/hparams_search/thesis_sanity.yaml`.
Runs land in `paths.yaml:new_runs_root`. The grid runs of the original experiments were
launched with `hparams_search` files that were later edited in place (`joint_slurm.yaml`,
`slurm_exp2.yaml`, `kuramoto.yaml`); the actual grids are recorded in each run's
`.hydra/overrides.yaml` in `cp_1`.
