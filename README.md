# CausalPretraining



<img src="visualization.png" alt="image" width="50%" height="auto">

This repository holds the official code for: [Embracing the black box: Heading towards foundation models for causal discovery from time series data](https://arxiv.org/abs/2402.09305) 

:exclamation: As the paper is still a preprint, we did not polish this repo fully. Most of the stuff work however just fine. We are going to revisit this in the future. :exclamation:

If you are simply interested in using one of our pretrained networks: [Usage](pretrained_model_example_usage.ipynb)

If you are interested in the data: [Here](data/generate_synthetic_ds.py)

If you are interested in reproducing the experimental results: All scripts are included and largely commented below. 
We provide the full code base here and most of the things are directly executable (with the exception of our synthetic experiments that we ran on a slurm cluster).



### Installation: 


The main environment can be installed with 

```
conda env create -f env_droplet.yml
```

Additionally, for PCMCI experiments, an additional environment can be installed via: 

```
conda env create -f env_tigramite_droplet.yml
```


### Usage

The project is loosely build upon (https://github.com/ashleve/lightning-hydra-template)


### DATA

To generate synthetic data samples run 

```bash
cd data
python generate_synthetic_ds.py --scale_up --synthetic_six --joint
```

To prepare other data sources used in the paper, download them here: 
```bash
wget -P data/ "https://github.com/anndvision/data/raw/main/jasmin/four_outputs_liqcf_pacific.csv"
wget -P data/ "https://raw.githubusercontent.com/wasimahmadpk/cdmi/refs/heads/main/datasets/river_discharge_data/data_dillingen.csv"
wget -P data/ "https://raw.githubusercontent.com/wasimahmadpk/cdmi/refs/heads/main/datasets/river_discharge_data/data_kempten.csv"
wget -P data/ "https://raw.githubusercontent.com/wasimahmadpk/cdmi/refs/heads/main/datasets/river_discharge_data/data_lenggries.csv"

```

For the Kuramoto data we use the generator of: https://github.com/loeweX/AmortizedCausalDiscovery

Then run: 
```bash
python generate_other_ds.py --kuramoto --aerosols --kuramoto_path path/to/download --aerosols_path path/to/download
```
to generate the proper formatting. The river dataset needs no formatting.

### Training
To simply train a default model with a small set of synthetic data samples run e.g.: 

```bash
python train.py model.model_type=transformer data.ds_name=SNL model=medium.yaml
```

Pretrained weights (Best Runs from the Joint Experiments and the size "big") are included in /pretrained_weights and can be used directly, e.g. as in [Make Graph](make_graphs.ipynb) or [Usage](pretrained_model_example_usage.ipynb) (Be careful, you need correlation injection inputs)


### Reproduce

All baselines included in the paper can be recreated by running e.g. :

```bash
python calc_baselines.py --corr --synth --var
```

You can find the summary display here  [Here](summarize_baseline_scorings.ipynb).
The results of the grid searches are provided in  [Here](summarize_cp_scorings.ipynb).
Further, zero-shot results as well as inference speeds for Causally Pretrained Neural Networks can be reproduced by running: 
```bash
python calc_cp_performance.py --rivers --aerosols --speed
```


Finally, the slurm scripts used for the grid search are included in  [slurm](/slurm). [This](calc_dist_preds.py) can be used to calculate a distribution_over_outputs (Appendix, Paper)


Feel free to contact me if you would like to have any additional content/information/code.  :sunglasses:



### Maintainers
[@GideonStein](https://github.com/Gideon-Stein).