Scripts in here were used to run grid search experiments on a Slurm cluster. 
They are not necessary to experiment with Causal Pretraining and are only here for archiving purposes.
ALL resulting scorings are included in /experimental_results as Pd tables.


The full process looked like this: 

```
python train.py -m hparams_search=full_search.yaml data.ds_name=base.yaml model=medium.yaml  
```
Runs a full search over all HP combinations and logs various things.


```
python slurm_calc_grid_scoring.py --data_path /path/to/slurm/results
```
This creates summary files with all successful runs (/experimental results, run_x)


```
python slurm_std_reruns.py -exp_path /path/to/summary/tables
```

Creates new jobs running multiple seeds for the best HP combinations.


```
python slurm_calc_final_scoring.py --data_path path/to/std/results
```
Summarizes the STD runs into a single table (/experimental results, std)


