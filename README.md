# Explainable Performance Distribution Prediction

Code and supplementary figures accompanying our paper on explaining algorithm performance distributions through interpretable statistics.

The approach predicts distributions of stochastic algorithm running times from instance features, then uses permutation importance and individual conditional expectation (ICE) curves to explain distribution statistics such as mean, variance, skewness, and kurtosis. Experiments compare an XGBoost-based predictor with DistNet on SAT solving and automated planning benchmarks.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `distnet.py` | Neural network predictor for distribution parameters, including training and model persistence. |
| `xgb_dist.py` | XGBoost-based distribution predictor with custom objectives and gradient stabilisation. |
| `distnet_tuner.py` | Hyperparameter optimisation using SMAC, with cross-validation or a train/validation split. |
| `run_distnet.py` | Benchmark loading, preprocessing, training, evaluation, permutation importance, ICE curves, and Submitit/Slurm job submission. |
| `supplementary_figures/` | Inverse Gaussian prediction and timing results promised in the paper. |

## Requirements and setup

The code uses Python 3.10+ and the [Algorithm Selection Framework (ASF)](https://github.com/hadarshavit/asf), together with PyTorch, XGBoost, NumPy, pandas, SciPy, scikit-learn, SMAC3, ConfigSpace, Plotly, and Submitit. PDF figure export also requires a compatible Plotly image-export setup.

The experiment driver imports `DistNet`, `XGBDistNet`, `tune_distnet`, and `NGBDistNet` from `asf.epm`; installing this repository alone does not make the local model files available through that package. Use an ASF installation containing the required implementations. The local model files also depend on ASF utilities.

Before running experiments:

1. Provide the benchmark runtime and feature data. `get_data_dir()` and `get_sc_dict()` in `run_distnet.py` define the expected locations and scenario names.
2. Update the machine-specific data, model, results, and figure paths in `run_distnet.py`.
3. For cluster execution, adapt the Slurm settings and experiment grid in its main block. The launcher also reads existing result CSVs to track completed experiments.

The benchmark inputs, trained models, and raw result CSVs are not bundled here. The driver is an experiment script with configured paths, rather than a command-line application.

## Running a single experiment

After installing dependencies and configuring paths, call `run_fold` from the repository directory to run one fold without the Slurm launcher:

```python
from run_distnet import run_fold

run_fold(
    scenario="spear_swgcp",
    fold=0,
    model_name="xgb_dist",
    dist_name="lognorm",
    num_train_samples=100,
    ncpus=8,
)
```

The driver uses 10-fold cross-validation over instances and evaluates negative log-likelihood, Kolmogorov–Smirnov (KS) distance, mass distance, and continuous ranked probability score (CRPS). It also computes permutation importance and ICE curves for predicted distribution statistics.

## Supplementary figures

The [supplementary figures](supplementary_figures/) complement the paper's log-normal results with inverse Gaussian results for all seven scenarios: Clasp-Factoring, LPG-Zenotravel, Saps-CV-VAR, Spear-QCP, Spear-SWGCP, YaLSAT-QCP, and YaLSAT-SWGCP.

Filenames follow `<scenario>_invgauss_<metric>_vs_obs.pdf`:

| Metric | Meaning |
| --- | --- |
| `val_ks_d` | KS distance between predicted and observed running time distributions |
| `val_nllh` | Negative log-likelihood |
| `val_crps` | Continuous ranked probability score |
| `val_mass` | Mass distance |
| `cpu_training_time` | CPU time for the final model fit, excluding hyperparameter optimisation and explainability analysis |

The horizontal axis gives the number of training observations per instance; lower values on the vertical axis are better. See the [shared legend](supplementary_figures/legend.pdf) for model colours. These 35 result PDFs and the legend are existing figures copied unchanged, without retraining or regeneration.
