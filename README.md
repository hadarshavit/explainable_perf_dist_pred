
This repository contains the implementation code for the paper "Explainable Predictions of Algorithm Performance Distributions" submitted to IJCAI-ECAI 2026

## Overview

This codebase implements methods for predicting probability distributions of algorithm runtimes using machine learning. The code includes implementations of:

- **DistNet**: A neural network-based approach for distributional prediction
- **XGBDistNet**: An XGBoost-based gradient boosting approach for distributional prediction  
- Hyperparameter tuning using SMAC
- Comprehensive evaluation metrics and analysis tools

## File Descriptions

### Core Model Files

- **`distnet.py`**: Implementation of the DistNet neural network model for predutional predictions. Includes model architecture, training procedures, and model persistence (save/load functionality).

- **`xgb_dist.py`**: Implementation of XGBDistNet, an XGBoost-based distributional predictor. Includes custom objective functions, gradient stabilization techniques, and optimal starting value initialization.

- **`distnet_tuner.py`**: Hyperparameter optimization module using SMAC (Sequential Model-based Algorithm Configuration). Supports both cross-validation and train-validation split modes for model selection.

### Experimental Code

- **`run_distnet.py`**: Main experimental script containing:
  - Data loading and preprocessing utilities
  - Training and evaluation pipelines
  - Statistical metrics computation (negative log-likelihood, KS test, CRPS, etc.)
  - Explainability analysis (permutation importance, ICE curves)
  - Batch job submission for cluster computing

## Requirements

The code requires the following main dependencies:

- Python 3.10+
- PyTorch
- XGBoost
- NumPy
- Pandas
- scikit-learn
- SciPy
- SMAC3 (for hyperparameter optimization)
- ConfigSpace
- Plotly (for visualization)
- submitit (for cluster job submission)
- asf
