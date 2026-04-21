import json
import logging
import lzma
import os
import pickle
import resource
import time
from functools import partial

import numpy as np
import pandas as pd
import submitit
import plotly.express as px
import torch
from asf.epm import DistNet, XGBDistNet, tune_distnet, NGBDistNet
from scipy.stats import (
    beta,
    betaprime,
    cauchy,
    expon,
    gamma,
    invgauss,
    levy,
    lognorm,
    lomax,
    norm,
    weibull_min,
)
from sklearn.model_selection import KFold
from smac import BlackBoxFacade, Callback
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue
import lzma

import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

def compute_skewness_lognorm(preds):
    shape = preds[:, 0]
    skewness = (np.exp(shape**2) + 2) * np.sqrt(np.exp(shape**2) - 1)
    return skewness

def compute_skewness_weibull(preds):
    from scipy.special import gamma as gamma_func
    c = preds[:, 0]
    g1 = gamma_func(1 + 1/c)
    g2 = gamma_func(1 + 2/c)
    g3 = gamma_func(1 + 3/c)
    mu = g1
    sigma = np.sqrt(g2 - g1**2)
    skewness = (g3 - 3*mu*sigma**2 - mu**3) / (sigma**3)
    return skewness

def compute_skewness_gamma(preds):
    a = preds[:, 0]
    skewness = 2 / np.sqrt(a)
    return skewness

def compute_skewness_invgauss(preds):
    mu = preds[:, 0]
    scale = preds[:, 1]
    lam = scale / mu
    skewness = 3 * np.sqrt(mu / lam)
    return skewness

def compute_skewness_exp(preds):
    return np.full(preds.shape[0], 2.0)

def compute_skewness_norm(preds):
    return np.full(preds.shape[0], 0.0)

def get_skewness_function(dist_name):
    skewness_functions = {
        "lognorm": compute_skewness_lognorm,
        "weibull": compute_skewness_weibull,
        "gamma": compute_skewness_gamma,
        "invgauss": compute_skewness_invgauss,
        "exp": compute_skewness_exp,
        "norm": compute_skewness_norm,
    }
    return skewness_functions[dist_name]

def compute_variance_lognorm(preds):
    shape = preds[:, 0]
    scale = preds[:, 1]
    variance = (np.exp(shape**2) - 1) * np.exp(2 * np.log(scale) + shape**2)
    return variance

def compute_variance_weibull(preds):
    from scipy.special import gamma as gamma_func
    c = preds[:, 0]
    scale = preds[:, 1]
    g1 = gamma_func(1 + 1/c)
    g2 = gamma_func(1 + 2/c)
    variance = scale**2 * (g2 - g1**2)
    return variance

def compute_variance_gamma(preds):
    a = preds[:, 0]
    scale = preds[:, 1]
    variance = a * scale**2
    return variance

def compute_variance_invgauss(preds):
    mu = preds[:, 0]
    scale = preds[:, 1]
    lam = scale / mu
    variance = mu**3 / lam
    return variance

def compute_variance_exp(preds):
    scale = preds[:, 0]
    variance = scale**2
    return variance

def compute_variance_norm(preds):
    scale = preds[:, 1]
    variance = scale**2
    return variance

def get_variance_function(dist_name):
    variance_functions = {
        "lognorm": compute_variance_lognorm,
        "weibull": compute_variance_weibull,
        "gamma": compute_variance_gamma,
        "invgauss": compute_variance_invgauss,
        "exp": compute_variance_exp,
        "norm": compute_variance_norm,
    }
    return variance_functions[dist_name]

def compute_cv_lognorm(preds):
    shape = preds[:, 0]
    cv = np.sqrt(np.exp(shape**2) - 1)
    return cv

def compute_cv_weibull(preds):
    from scipy.special import gamma as gamma_func
    c = preds[:, 0]
    g1 = gamma_func(1 + 1/c)
    g2 = gamma_func(1 + 2/c)
    cv = np.sqrt(g2 / g1**2 - 1)
    return cv

def compute_cv_gamma(preds):
    a = preds[:, 0]
    cv = 1 / np.sqrt(a)
    return cv

def compute_cv_invgauss(preds):
    mu = preds[:, 0]
    scale = preds[:, 1]
    lam = scale / mu
    cv = np.sqrt(mu / lam)
    return cv

def compute_cv_exp(preds):
    return np.full(preds.shape[0], 1.0)

def compute_cv_norm(preds):
    mu = preds[:, 0]
    scale = preds[:, 1]
    cv = scale / np.abs(mu)
    return cv

def get_cv_function(dist_name):
    cv_functions = {
        "lognorm": compute_cv_lognorm,
        "weibull": compute_cv_weibull,
        "gamma": compute_cv_gamma,
        "invgauss": compute_cv_invgauss,
        "exp": compute_cv_exp,
        "norm": compute_cv_norm,
    }
    return cv_functions[dist_name]

def compute_kurtosis_lognorm(preds):
    shape = preds[:, 0]
    s2 = shape**2
    kurtosis = np.exp(4*s2) + 2*np.exp(3*s2) + 3*np.exp(2*s2) - 6
    return kurtosis

def compute_kurtosis_weibull(preds):
    from scipy.special import gamma as gamma_func
    c = preds[:, 0]
    g1 = gamma_func(1 + 1/c)
    g2 = gamma_func(1 + 2/c)
    g3 = gamma_func(1 + 3/c)
    g4 = gamma_func(1 + 4/c)
    mu = g1
    sigma2 = g2 - g1**2
    kurtosis = (g4 - 4*g3*mu + 6*g2*mu**2 - 3*mu**4) / sigma2**2 - 3
    return kurtosis

def compute_kurtosis_gamma(preds):
    a = preds[:, 0]
    kurtosis = 6 / a
    return kurtosis

def compute_kurtosis_invgauss(preds):
    mu = preds[:, 0]
    scale = preds[:, 1]
    lam = scale / mu
    kurtosis = 15 * mu / lam
    return kurtosis

def compute_kurtosis_exp(preds):
    return np.full(preds.shape[0], 6.0)

def compute_kurtosis_norm(preds):
    return np.full(preds.shape[0], 0.0)

def get_kurtosis_function(dist_name):
    kurtosis_functions = {
        "lognorm": compute_kurtosis_lognorm,
        "weibull": compute_kurtosis_weibull,
        "gamma": compute_kurtosis_gamma,
        "invgauss": compute_kurtosis_invgauss,
        "exp": compute_kurtosis_exp,
        "norm": compute_kurtosis_norm,
    }
    return kurtosis_functions[dist_name]

def compute_iqr_lognorm(preds):
    shape = preds[:, 0]
    scale = preds[:, 1]
    from scipy.stats import norm as norm_dist
    q1 = scale * np.exp(shape * norm_dist.ppf(0.25))
    q3 = scale * np.exp(shape * norm_dist.ppf(0.75))
    return q3 - q1

def compute_iqr_weibull(preds):
    c = preds[:, 0]
    scale = preds[:, 1]
    q1 = scale * (-np.log(1 - 0.25))**(1/c)
    q3 = scale * (-np.log(1 - 0.75))**(1/c)
    return q3 - q1

def compute_iqr_gamma(preds):
    from scipy.stats import gamma as gamma_dist
    a = preds[:, 0]
    scale = preds[:, 1]
    iqr = np.array([gamma_dist.ppf(0.75, a=a_i, scale=s_i) - gamma_dist.ppf(0.25, a=a_i, scale=s_i) 
                    for a_i, s_i in zip(a, scale)])
    return iqr

def compute_iqr_invgauss(preds):
    from scipy.stats import invgauss as invgauss_dist
    mu = preds[:, 0]
    scale = preds[:, 1]
    iqr = np.array([invgauss_dist.ppf(0.75, mu=mu_i/s_i, scale=s_i) - 
                    invgauss_dist.ppf(0.25, mu=mu_i/s_i, scale=s_i) 
                    for mu_i, s_i in zip(mu, scale)])
    return iqr

def compute_iqr_exp(preds):
    scale = preds[:, 0]
    q1 = -scale * np.log(1 - 0.25)
    q3 = -scale * np.log(1 - 0.75)
    return q3 - q1

def compute_iqr_norm(preds):
    from scipy.stats import norm as norm_dist
    scale = preds[:, 1]
    iqr = 2 * scale * norm_dist.ppf(0.75)
    return iqr

def get_iqr_function(dist_name):
    iqr_functions = {
        "lognorm": compute_iqr_lognorm,
        "weibull": compute_iqr_weibull,
        "gamma": compute_iqr_gamma,
        "invgauss": compute_iqr_invgauss,
        "exp": compute_iqr_exp,
        "norm": compute_iqr_norm,
    }
    return iqr_functions[dist_name]

def compute_mean_lognorm(preds):
    shape = preds[:, 0]
    scale = preds[:, 1]
    mean = np.exp(np.log(scale) + shape**2 / 2)
    return mean

def compute_mean_weibull(preds):
    from scipy.special import gamma as gamma_func
    c = preds[:, 0]
    scale = preds[:, 1]
    mean = scale * gamma_func(1 + 1/c)
    return mean

def compute_mean_gamma(preds):
    a = preds[:, 0]
    scale = preds[:, 1]
    mean = a * scale
    return mean

def compute_mean_invgauss(preds):
    mu = preds[:, 0]
    return mu

def compute_mean_exp(preds):
    scale = preds[:, 0]
    return scale

def compute_mean_norm(preds):
    mu = preds[:, 0]
    return mu

def get_mean_function(dist_name):
    mean_functions = {
        "lognorm": compute_mean_lognorm,
        "weibull": compute_mean_weibull,
        "gamma": compute_mean_gamma,
        "invgauss": compute_mean_invgauss,
        "exp": compute_mean_exp,
        "norm": compute_mean_norm,
    }
    return mean_functions[dist_name]

def get_property_function(property_name, dist_name):
    if property_name == "skewness":
        return get_skewness_function(dist_name)
    elif property_name == "variance":
        return get_variance_function(dist_name)
    elif property_name == "cv":
        return get_cv_function(dist_name)
    elif property_name == "mean":
        return get_mean_function(dist_name)
    elif property_name == "kurtosis":
        return get_kurtosis_function(dist_name)
    elif property_name == "iqr":
        return get_iqr_function(dist_name)
    else:
        raise ValueError(f"Unknown property: {property_name}")

class PropertyWrapper:
    def __init__(self, model, property_func):
        self.model = model
        self.property_func = property_func

    def fit(self, X, y):
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()
        result = self.property_func(preds)
        if hasattr(result, 'numpy'):
            result = result.numpy()
        return result

def compute_permutation_importance_local(model, X, y_property, n_repeats=10, random_state=42):
    def property_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        if hasattr(y_pred, 'numpy'):
            y_pred = y_pred.numpy()
        if hasattr(y, 'numpy'):
            y = y.numpy()
        return -np.mean((y - y_pred) ** 2)

    result = permutation_importance(
        model, X, y_property,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=property_scorer
    )
    return result

def compute_ice_curves_local(model, X, feature_idx, property_func, num_grid_points=50, norm_params=None):
    feature_values = X[:, feature_idx]
    grid_min = np.min(feature_values)
    grid_max = np.max(feature_values)
    grid_values_normalized = np.linspace(grid_min, grid_max, num_grid_points)
    
    ice_curves = np.zeros((X.shape[0], num_grid_points))
    
    sample_idx = 0
    sample_preds = []
    
    for i, grid_val in enumerate(grid_values_normalized):
        X_modified = X.copy()
        X_modified[:, feature_idx] = grid_val
        preds = model.predict(X_modified)
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()
        ice_curves[:, i] = property_func(preds)
        
        if sample_idx < preds.shape[0]:
            sample_preds.append(preds[sample_idx])
    
    pdp_curve = np.mean(ice_curves, axis=0)
    
    if norm_params is not None:
        if norm_params['method'] == 'meanstd':
            grid_values_display = grid_values_normalized * norm_params['std'][feature_idx] + norm_params['mean'][feature_idx]
        elif norm_params['method'] == 'minmax':
            grid_values_display = grid_values_normalized * norm_params['max'][feature_idx] + norm_params['min'][feature_idx]
        else:
            grid_values_display = grid_values_normalized
    else:
        grid_values_display = grid_values_normalized
    
    property_range = ice_curves.max() - ice_curves.min()
    property_std = ice_curves.std()
    property_mean = ice_curves.mean()
    
    if len(sample_preds) > 0:
        param_ranges = []
        for param_idx in range(sample_preds[0].shape[0] if len(sample_preds[0].shape) > 0 else 1):
            if len(sample_preds[0].shape) > 0:
                param_vals = [p[param_idx] for p in sample_preds]
            else:
                param_vals = sample_preds
            param_ranges.append((np.min(param_vals), np.max(param_vals), np.max(param_vals) - np.min(param_vals)))
    
    print(f"  ICE diagnostics - Feature idx {feature_idx}:")
    print(f"    Grid range (normalized): [{grid_min:.3f}, {grid_max:.3f}]")
    print(f"    Grid range (original): [{grid_values_display[0]:.3f}, {grid_values_display[-1]:.3f}]")
    print(f"    Property Y-values: mean={property_mean:.6f}, range={property_range:.6f}, std={property_std:.6f}")
    print(f"    Relative variation: {(property_range / property_mean * 100):.4f}% of mean")
    if len(sample_preds) > 0:
        for idx, (pmin, pmax, prange) in enumerate(param_ranges):
            print(f"    Sample prediction param[{idx}] range: [{pmin:.6f}, {pmax:.6f}] (Δ={prange:.6f})")
    
    if property_range < 1e-10:
        print(f"    WARNING: Property values are essentially constant (range < 1e-10)")
        print(f"    This means the model prediction for this property does NOT change with this feature")
    
    return grid_values_display, ice_curves, pdp_curve

def plot_permutation_importance_local(importance_result, feature_names, output_path, top_k=5, property_name="Property"):
    sorted_idx = importance_result.importances_mean.argsort()[::-1][:top_k]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[feature_names[i] for i in sorted_idx],
        x=importance_result.importances_mean[sorted_idx],
        orientation='h',
    ))
    
    fig.update_layout(
        width=125,
        height=125,
        title=None,
        xaxis=dict(
            title=None,
            tickfont=dict(size=5)
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(size=5),
            tickangle=-45
        ),
        margin=dict(l=20, r=0, t=0, b=0),
        showlegend=False
    )
    
    fig.write_image(output_path)

def plot_ice_curves(grid_values, ice_curves, pdp_curve, feature_name, output_path, property_name="Property"):
    fig = go.Figure()
    
    n_curves = min(ice_curves.shape[0], 100)
    sample_indices = np.random.choice(ice_curves.shape[0], n_curves, replace=False)

    ice_curves_centered = ice_curves - ice_curves[:, 0:1]
    pdp_curve_centered = pdp_curve - pdp_curve[0]

    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=grid_values,
            y=ice_curves_centered[idx],
            mode='lines',
            line=dict(color=px.colors.qualitative.Plotly[7], width=0.5),
            opacity=0.5,
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=grid_values,
        y=pdp_curve_centered,
        mode='lines',
        line=dict(color='red', width=2),
        name='PDP'
    ))

    fig.add_hline(y=0, line_dash='dash', line_color="grey", opacity=0.5)

    fig.update_layout(
        width=125,
        height=125,
        title=None,
        xaxis=dict(
            title=dict(text=feature_name, font=dict(size=5)),
            tickfont=dict(size=7),
            title_standoff=5
        ),
        yaxis=dict(
            title=dict(text=f'Δ {property_name}', font=dict(size=5)),
            tickfont=dict(size=5)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    fig.write_image(output_path)

def set_up_logging(ncpus=1):
    logging.basicConfig(level=logging.INFO)
    torch.set_num_interop_threads(1)
    torch.set_num_threads(ncpus)

def _get_dists_dict():
    from asf.predictors.utils import losses as _losses

    return {
        "lognorm": (
            lognorm,
            _losses.lognorm_loss,
            _losses.LOGNORM_N_PARAMS,
            ["s"],
            True,
        ),
        "exp": (expon, _losses.exp_loss, _losses.EXP_N_PARAMS, [], True),
        "weibull": (
            weibull_min,
            _losses.weibull_loss,
            _losses.WEIBULL_N_PARAMS,
            ["c"],
            True,
        ),
        "invgauss": (
            invgauss,
            _losses.invgauss_loss,
            _losses.INVGAUSS_N_PARAMS,
            ["mu"],
            True,
        ),
        "gamma": (gamma, _losses.gamma_loss, _losses.GAMMA_N_PARAMS, ["a"], True),
        "cauchy": (cauchy, _losses.cauchy_loss, _losses.CAUCHY_N_PARAMS, [], True),
        "levy": (levy, _losses.levy_loss, _losses.LEVY_N_PARAMS, [], True),
        "beta": (beta, _losses.beta_loss, _losses.BETA_N_PARAMS, ["a", "b"], True),
        "beta_prime": (
            betaprime,
            _losses.betaprime_loss,
            _losses.BETAPRIME_N_PARAMS,
            ["a", "b"],
            True,
        ),
        "lomax": (lomax, _losses.lomax_loss, _losses.LOMAX_N_PARAMS, ["c"], True),
        "norm": (norm, _losses.normal_loss, _losses.NORM_N_PARAMS, [], False),
    }

class StopCallback(Callback):
    def __init__(self, stop_after: int):
        self._stop_after = stop_after

    def on_tell_end(
        self, smbo: SMBO, info: TrialInfo, value: TrialValue
    ) -> bool | None:
        if smbo.runhistory.finished == self._stop_after:
            return False

        return None

def get_data_dir():
    return "/storage/work/anonymous/dist_explainability/DistNetData"

def get_sc_dict():
    data_dir = get_data_dir()
    sc_dict = {
        "clasp_factoring": {
            "scen": "clasp-3.0.4-p8_rand_factoring",
            "features": "%s/clasp-3.0.4-p8_rand_factoring/features.txt" % data_dir,
            "domain": "sat",
            "use": ("SAT",),
            "cutoff": 5000,
        },
        "saps-CVVAR": {
            "scen": "CP06_CV-VAR",
            "features": "%s/CP06_CV-VAR/features.txt" % data_dir,
            "domain": "sat",
            "use": ("SAT",),
            "cutoff": 60,
        },
        "spear_qcp": {
            "scen": "spear_qcp-hard",
            "features": "%s/spear_qcp-hard/features.txt" % data_dir,
            "domain": "sat",
            "use": ("SAT",),
            "cutoff": 5000,
        },
        "yalsat_qcp": {
            "scen": "yalsat_qcp-hard",
            "features": "%s/yalsat_qcp-hard/features.txt" % data_dir,
            "domain": "sat",
            "use": ("SAT",),
            "cutoff": 5000,
        },
        "spear_swgcp": {
            "scen": "spear_smallworlds",
            "features": "%s/spear_smallworlds/features.txt" % data_dir,
            "domain": "sat",
            "use": ("SAT",),
            "cutoff": 5000,
        },
        "yalsat_swgcp": {
            "scen": "yalsat_smallworlds",
            "features": "%s/yalsat_smallworlds/features.txt" % data_dir,
            "domain": "sat",
            "use": ("SAT",),
            "cutoff": 5000,
        },
        "lpg-zeno": {
            "scen": "lpg-zenotravel",
            "features": "%s/lpg-zenotravel/features.txt" % data_dir,
            "domain": "planning",
            "use": ("SAT",),
            "cutoff": 300,
        },
    }
    return sc_dict

def read_results(data_dir, cutoff=300, runs_per_inst=100, suffix="train"):
    fl_name = (
        "%s/validate-random-%s/validationRunResultLineMatrix-cli-1-"
        "walltimeworker.csv" % (data_dir, suffix)
    )

    if not os.path.exists(fl_name):
        raise ValueError("%s does not exist" % fl_name)

    tmp_data = list()
    inst_ls = list()
    sat_ls = list()
    with open(fl_name, "r") as fl:
        fl.readline()
        for line in fl:
            line = line.strip().replace('"', "").split(",")
            tmp_data.append(min(cutoff, float(line[3].strip())))
            sat_ls.append(str(line[2]))
            inst_ls.append(line[0])

    data = list()
    sat_data = list()
    for inst in range(int(len(tmp_data) / runs_per_inst)):
        data.append(tmp_data[inst * runs_per_inst : (inst + 1) * runs_per_inst])
        sat_data.append(sat_ls[inst * runs_per_inst : (inst + 1) * runs_per_inst])
    data = np.array(data)
    return data, inst_ls, sat_data

def load_features(fl_name):
    feat_dict = dict()
    feature_names = None
    lens = None
    with open(fl_name) as fh:
        header = fh.readline().strip()
        feature_names = header.split(",")[1:]
        for line in fh:
            line = line.strip().split(",")
            key = line[0]
            val = [float(i) for i in line[1:]]
            if lens is None:
                lens = len(val)

            if len(val) != lens:
                raise ValueError(
                    "Feature length mismatch for instance %s: expected %d, got %d"
                    % (key, lens, len(val))
                )
            else:
                pass
            feat_dict[key] = val
    return feat_dict, feature_names

def get_data(scenario, data_dir, sc_dict, retrieve=["SAT", "UNSAT"]):
    data_dir = data_dir + "/" + sc_dict[scenario]["scen"] + "/"
    runtimes, inst_ls, sat_ls = read_results(
        data_dir=data_dir,
        cutoff=sc_dict[scenario]["cutoff"],
        runs_per_inst=100,
        suffix="train",
    )
    try:
        test_runtimes, test_inst_ls, test_sat_ls = read_results(
            data_dir=data_dir,
            cutoff=sc_dict[scenario]["cutoff"],
            runs_per_inst=100,
            suffix="test",
        )
        runtimes = np.vstack([runtimes, test_runtimes])
        inst_ls.extend(test_inst_ls)
        sat_ls.extend(test_sat_ls)
    except ValueError:
        print("Could not find test data")

    try:
        feat_dict, feature_names = load_features(sc_dict[scenario]["features"])
    except TypeError:
        print("Features file %s does not exist" % sc_dict[scenario]["features"])
        feat_dict = dict((i, np.random.random_sample(2)) for i in inst_ls[::100])
        feature_names = [f"Feature_{i}" for i in range(2)]
    features = list()
    for i in inst_ls[::100]:
        features.append(feat_dict[i])

    features = np.array(features)
    runtimes = np.array(runtimes)

    runtimes, features, sat_ls = remove_instances_with_status(
        runningtimes=runtimes, features=features, sat_ls=sat_ls, status="CRASHED"
    )
    runtimes, features, sat_ls = remove_instances_with_status(
        runningtimes=runtimes, features=features, sat_ls=sat_ls, status="TIMEOUT"
    )
    runtimes, features, sat_ls = remove_timeouts(
        runningtimes=runtimes,
        features=features,
        cutoff=sc_dict[scenario]["cutoff"],
        sat_ls=sat_ls,
    )
    runtimes, features, sat_ls = remove_constant_instances(
        runningtimes=runtimes, features=features, sat_ls=sat_ls
    )

    if "SAT" not in retrieve:
        runtimes, features, sat_ls = remove_instances_with_status(
            runningtimes=runtimes, features=features, sat_ls=sat_ls, status="SAT"
        )
    if "UNSAT" not in retrieve:
        runtimes, features, sat_ls = remove_instances_with_status(
            runningtimes=runtimes, features=features, sat_ls=sat_ls, status="UNSAT"
        )

    features = feature_imputation(features, impute_val=-512, impute_with="median")
    return runtimes, features, sat_ls, feature_names

def remove_timeouts(runningtimes, cutoff, features=None, sat_ls=None):

    if features is None:
        features = [0] * runningtimes.shape[0]
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance >= cutoff):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    return np.array(new_rt), np.array(new_ft), new_sl

def remove_instances_with_status(runningtimes, features, sat_ls=None, status="CRASHED"):
    if sat_ls is None:
        print("Could not remove %s instances" % status)

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if status not in s:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)
    return np.array(new_rt), np.array(new_ft), new_sl

def remove_constant_instances(runningtimes, features, sat_ls=None):
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if np.std(f) > 0:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)

    return np.array(new_rt), np.array(new_ft), new_sl

def feature_imputation(features, impute_val=-512, impute_with="median"):
    if impute_with == "median":
        for col in range(features.shape[1]):
            med = np.median(features[:, col])
            features[:, col] = [med if i == impute_val else i for i in features[:, col]]
    return features

def compute_nllh(runtimes, preds, dist=lognorm, floc=False, scale=1.0):
    total_nllh = []

    for i, pred in enumerate(preds):
        r_i = np.asarray(runtimes[i])
        if floc:
            params = list(pred)
            params.insert(-1, 0.0)
            logpdf_vals = dist.logpdf(r_i, *params)
        else:
            logpdf_vals = dist.logpdf(r_i, *pred)
        total_nllh.append(-np.mean(logpdf_vals) + np.log(scale))

    return np.mean(total_nllh)

def compute_nllh_distnet(runtimes, preds, dist=lognorm, floc=False):
    total_nllh = []

    for i, pred in enumerate(preds):
        r_i = np.asarray(runtimes[i])
        if floc:
            params = list(pred)
            params.insert(-1, 0.0)
            logpdf_vals = dist.logpdf(r_i, *params)
        else:
            logpdf_vals = dist.logpdf(r_i, *pred)

        nllh_per_inst = logpdf_vals + np.log(np.max(r_i))
        total_nllh.append(np.mean(-nllh_per_inst))

    return np.mean(total_nllh)

def _scipy_params_adjust(params, dist_scipy=None, floc=False):
    params = list(params)
    if dist_scipy is not None:
        shape_names = dist_scipy.shapes
        num_shape_args = 0
        if shape_names:
            num_shape_args = len(shape_names.split(","))
    else:
        num_shape_args = 0

    if floc:
        if len(params) == num_shape_args + 1:
            params.insert(-1, 0.0)
    return params
    return params

def get_ks_for_instance(dist_scipy, params, instance, floc=False):
    from scipy.stats import ks_2samp

    params = _scipy_params_adjust(params, dist_scipy, floc=floc)
    n_samp = max(200, len(instance))
    try:
        samps = dist_scipy.rvs(*params, size=n_samp)
    except Exception:
        samps = dist_scipy.rvs(*params, size=min(50, n_samp))
    stat, p = ks_2samp(instance, samps)
    return stat, p

def get_mass_for_instance(dist_scipy, params, instance, floc=False):
    params = _scipy_params_adjust(params, dist_scipy, floc=floc)
    MIN_T = 0.0
    MAX_T = max(instance) * 1.5
    inside_mass = dist_scipy.cdf(MAX_T, *params) - dist_scipy.cdf(MIN_T, *params)
    return 1.0 - inside_mass

def crps_sample_based(dist_scipy, params, observed_runs, n_samples=200, floc=False):
    params = _scipy_params_adjust(params, dist_scipy, floc=floc)
    s = dist_scipy.rvs(*params, size=n_samples)
    absdiff = np.mean(np.abs(s[None, :] - observed_runs[:, None]), axis=1)
    sample_pair_mean = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
    crps_vals = absdiff - sample_pair_mean
    return np.mean(crps_vals)

def comput_skewness(preds, dist=lognorm):
    skewness_vals = []

    for i, pred in enumerate(preds):
        shape = pred[0]
        scale = pred[1]

        skewness = (np.exp(shape**2) + 2) * np.sqrt(np.exp(shape**2) - 1)
        skewness_vals.append(skewness)

    return np.mean(skewness_vals)

def remove_zeros(runningtimes, features=None, sat_ls=None):

    if features is None:
        features = [0] * runningtimes.shape[0]
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance <= 0):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    return np.array(new_rt), np.array(new_ft), new_sl

def det_constant_features(X):
    max_ = X.max(axis=0)
    min_ = X.min(axis=0)
    diff = max_ - min_

    det_idx = np.where(diff <= 10e-10)
    return det_idx

def det_transformation(X):
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0) - min_
    return min_, max_

def preprocess_features(tra_X, val_X, scal="meanstd"):

    del_idx = det_constant_features(tra_X)
    tra_X = np.delete(tra_X, del_idx, axis=1)
    val_X = np.delete(val_X, del_idx, axis=1)

    if scal == "minmax":
        min_, max_ = det_transformation(tra_X)
        tra_X = (tra_X - min_) / max_
        val_X = (val_X - min_) / max_
        norm_params = {'method': 'minmax', 'min': min_, 'max': max_, 'deleted_idx': del_idx}
    else:
        mean_ = tra_X.mean(axis=0)
        std_ = tra_X.std(axis=0)
        tra_X = (tra_X - mean_) / std_
        val_X = (val_X - mean_) / std_
        norm_params = {'method': 'meanstd', 'mean': mean_, 'std': std_, 'deleted_idx': del_idx}

    return tra_X, val_X, norm_params

def run_fold(
    scenario,
    fold,
    model_name,
    dist_name,
    num_train_samples,
    num_train_instances=None,
    ncpus=1,
):
    print(
        f"Running scenario {scenario} fold {fold} model {model_name} dist {dist_name} num_train_samples {num_train_samples} num_train_instances {num_train_instances}"
    )
    np.random.seed(2)
    torch.manual_seed(2)

    data_dir = get_data_dir()
    sc_dict = get_sc_dict()
    runtimes, features, sat_ls, feature_names_orig = get_data(
        scenario=scenario,
        data_dir=data_dir,
        sc_dict=sc_dict,
        retrieve=sc_dict[scenario]["use"],
    )

    dists_dict = _get_dists_dict()
    dist, loss_function, n_loss_params, param_names, floc = dists_dict[dist_name]

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    idx = list(range(runtimes.shape[0]))
    for cfold, (train, valid) in enumerate(kf.split(idx)):
        if cfold != fold:
            continue
        train_idx = train
        val_idx = valid

    val_features = features[val_idx]
    train_features = features[train_idx]
    train_runtimes = runtimes[train_idx]
    train_runtimes_full = train_runtimes
    val_runtimes = runtimes[val_idx]

    if num_train_instances is not None:
        if num_train_instances >= train_runtimes.shape[0]:
            return
        print(f"Downsampling to {num_train_instances} training instances")
        instance_idx = list(range(train_runtimes.shape[0]))
        np.random.shuffle(instance_idx)
        instance_idx = instance_idx[:num_train_instances]
        train_runtimes = train_runtimes[instance_idx]
        train_runtimes_full = train_runtimes
        train_features = train_features[instance_idx]

    if num_train_samples != 100:
        print(f"Subsampling to {num_train_samples} training samples")
        subset_idx = list(range(train_runtimes.shape[1]))
        np.random.shuffle(subset_idx)
        subset_idx = subset_idx[:num_train_samples]

        train_runtimes = train_runtimes[:, subset_idx]

    train_max = np.max(train_runtimes)
    train_runtimes_norm = train_runtimes / train_max
    val_runtimes_norm = val_runtimes / train_max

    train_features, val_features, norm_params = preprocess_features(
        tra_X=train_features, val_X=val_features, scal="meanstd"
    )

    tune = True
    if model_name == "distnet":
        model = DistNet
        kwargs = {
            "epochs": 1000,
            "loss_function": loss_function,
            "n_loss_params": n_loss_params,
        }
    elif model_name == "xgb_dist":
        model = XGBDistNet

        kwargs = {
            "batch_size": 250_000_000,
            "disable_default_eval_metric": True,
            "base_score": 0.0,
            "loss_function": loss_function,
            "n_loss_params": n_loss_params,
            "early_stopping_rounds": 10,
            "early_stopping_tolerance": 1e-6,
            "n_jobs": ncpus,
            "verbosity": 3,
        }
    elif model_name == "ngb_dist":
        model = NGBDistNet

        kwargs = {
            "distribution": dist_name,
        }
    elif model_name == "distnet_default":
        model = DistNet.get_from_configuration(
            configuration=DistNet.get_configuration_space().get_default_configuration(),
            input_size=train_features.shape[1],
            output_size=n_loss_params,
        )
        tune = False

    instances_suffix = (
        f"_inst{num_train_instances}" if num_train_instances is not None else ""
    )

    print(
        f"Training features shape: {train_features.shape}, runtimes shape: {train_runtimes.shape}"
    )

    if tune:
        print(f"Tuning hyperparameters with SMAC... Output: /home/anonymous/dist_explainability/results/smac/{scenario}_{model_name}_{dist_name}_{num_train_samples}{instances_suffix}_fold{fold}")
        model = tune_distnet(
            model=model,
            X=train_features,
            y=train_runtimes_norm,
            output_size=n_loss_params,
            validation_mode="trainval",
            seed=2,
            features_preprocessing=None,
            cv=10,
            timeout=np.inf,
            metric=partial(compute_nllh, dist=dist, floc=floc, scale=train_max),
            runcount_limit=100,
            output_dir=f"/home/anonymous/dist_explainability/results/smac/{scenario}_{model_name}_{dist_name}_{num_train_samples}{instances_suffix}_fold{fold}",
            distnet_kwargs=kwargs,
            smac_kwargs={
                "callbacks": [
                    StopCallback(stop_after=3600 * 120)
                ],
            },
            smac_scenario_kwargs={
                "name": f"{scenario}_{model_name}_{dist_name}_{num_train_samples}{instances_suffix}_fold{fold}",
                "crash_cost": 1e12,
                "use_default_config": False,
            },
        )

    model = model()
    print(train_features.shape, train_runtimes_norm.shape)

    start = time.time()
    start_cpu = time.process_time()
    model.fit(X=train_features, y=train_runtimes_norm)
    end_cpu = time.process_time()
    end = time.time()

    training_time = end - start
    cpu_time = end_cpu - start_cpu
    print(
        f"Training time for {model_name} on scenario {scenario} fold {fold}: {end - start:.2f} seconds"
    )
    print(
        f"CPU training time for {model_name} on scenario {scenario} fold {fold}: {cpu_time:.2f} seconds"
    )

    start = time.time()
    start_cpu = time.process_time()
    preds = model.predict(val_features)
    end_cpu = time.process_time()
    end = time.time()

    inference_time = end - start
    cpu_inference_time = end_cpu - start_cpu
    print(
        f"Inference time for {model_name} on scenario {scenario} fold {fold}: {end - start:.2f} seconds"
    )
    print(
        f"CPU inference time for {model_name} on scenario {scenario} fold {fold}: {cpu_inference_time:.2f} seconds"
    )

    ks_stats = []
    ks_ps = []
    mass_vals = []
    crps_vals = []
    for i, pred in enumerate(preds):
        r_i = val_runtimes_norm[i]
        try:
            stat, p = get_ks_for_instance(dist, pred, r_i, floc=floc)
        except Exception:
            stat, p = np.nan, np.nan
        try:
            mass = get_mass_for_instance(dist, pred, r_i, floc=floc)
        except Exception:
            mass = np.nan
        try:
            crps = crps_sample_based(dist, pred, r_i, n_samples=200, floc=floc)
        except Exception:
            crps = np.nan

        ks_stats.append(stat)
        ks_ps.append(p)
        mass_vals.append(mass)
        crps_vals.append(crps)

    val_ks_d = np.nanmean(ks_stats) if len(ks_stats) > 0 else np.nan
    val_ks_p = np.nanmean(
        [1 if (p is not None and not np.isnan(p) and p < 0.01) else 0 for p in ks_ps]
    )
    val_mass = np.nanmean(mass_vals) if len(mass_vals) > 0 else np.nan
    val_crps = np.nanmean(crps_vals) if len(crps_vals) > 0 else np.nan

    total_nllh = compute_nllh(
        val_runtimes_norm, preds, dist=dist, floc=floc, scale=train_max
    )
    total_nllh_norm = compute_nllh(
        val_runtimes_norm, preds, dist=dist, floc=floc, scale=1.0
    )
    total_nllh_distnet = compute_nllh_distnet(
        val_runtimes_norm, preds, dist=dist, floc=floc
    )

    print(f"{model_name} Val NLLH: {np.mean(total_nllh):.4f}")

    fitted_params = []
    for i, pred in enumerate(preds):
        r_i = val_runtimes_norm[i]

        params = dist.fit(r_i, floc=0 if floc else None)
        fitted_params.append(params)

    total_fitted_nllh = compute_nllh(
        val_runtimes_norm, fitted_params, dist=dist, floc=False, scale=train_max
    )
    total_nllh_distnet_fitted = compute_nllh_distnet(
        val_runtimes_norm, fitted_params, dist=dist, floc=False
    )

    fitted_params = []
    for i, pred in enumerate(preds):
        r_i = val_runtimes[i]

        params = dist.fit(r_i, floc=0 if floc else None)
        fitted_params.append(params)

    total_fitted_nllh_orig = compute_nllh(
        val_runtimes, fitted_params, dist=dist, floc=False, scale=1.0
    )
    total_nllh_distnet_fitted_orig = compute_nllh_distnet(
        val_runtimes, fitted_params, dist=dist, floc=False
    )

    train_preds = model.predict(train_features)

    train_runtimes_full_norm = train_runtimes_full / train_max
    train_full_total_nllh = compute_nllh(
        train_runtimes_full_norm, train_preds, dist=dist, floc=floc, scale=train_max
    )
    train_full_total_nllh_distnet = compute_nllh_distnet(
        train_runtimes_full_norm, train_preds, dist=dist, floc=floc
    )
    train_full_total_nllh_norm = compute_nllh(
        train_runtimes_full_norm, train_preds, dist=dist, floc=floc, scale=1.0
    )
    train_full_total_nllh_distnet_norm = compute_nllh_distnet(
        train_runtimes_full_norm, train_preds, dist=dist, floc=floc
    )

    train_total_nllh = compute_nllh(
        train_runtimes_norm, train_preds, dist=dist, floc=floc, scale=train_max
    )
    train_total_nllh_distnet = compute_nllh_distnet(
        train_runtimes_norm, train_preds, dist=dist, floc=floc
    )
    train_total_nllh_norm = compute_nllh(
        train_runtimes_norm, train_preds, dist=dist, floc=floc, scale=1.0
    )
    train_total_nllh_distnet_norm = compute_nllh_distnet(
        train_runtimes_norm, train_preds, dist=dist, floc=floc
    )

    print(f"{model_name} Train NLLH: {np.mean(train_total_nllh):.4f}")

    results = [
        {
            "fold": fold,
            "scenario": scenario,
            "model": model_name,
            "dist": dist_name,
            "num_train_samples": num_train_samples,
            "num_train_instances": num_train_instances
            if num_train_instances is not None
            else train_runtimes.shape[0],
            "val_nllh": np.mean(total_nllh),
            "val_nllh_orig": np.mean(total_nllh),
            "val_nllh_norm": np.mean(total_nllh_norm),
            "val_fitted_nllh": np.mean(total_fitted_nllh),
            "val_fitted_nllh_orig": np.mean(total_fitted_nllh_orig),
            "train_nllh": np.mean(train_total_nllh),
            "train_nllh_orig": np.mean(train_total_nllh),
            "train_nllh_norm": np.mean(train_total_nllh_norm),
            "train_full_nllh": np.mean(train_full_total_nllh),
            "train_full_nllh_orig": np.mean(train_full_total_nllh),
            "train_full_nllh_norm": np.mean(train_full_total_nllh_norm),
            "training_time": training_time,
            "inference_time": inference_time,
            "cpu_training_time": cpu_time,
            "cpu_inference_time": cpu_inference_time,
            "val_nllh_distnet": np.mean(total_nllh_distnet),
            "val_nllh_distnet_fitted": np.mean(total_nllh_distnet_fitted_orig),
            "train_nllh_distnet": np.mean(train_total_nllh_distnet),
            "train_full_nllh_distnet": np.mean(train_full_total_nllh_distnet),
            "val_nllh_distnet_norm": np.mean(total_nllh_distnet),
            "val_nllh_distnet_fitted_norm": np.mean(total_nllh_distnet_fitted),
            "train_nllh_distnet_norm": np.mean(train_total_nllh_distnet_norm),
            "train_full_nllh_distnet_norm": np.mean(train_full_total_nllh_distnet_norm),
            "train_max": train_max,
            "val_ks_p": val_ks_p,
            "val_ks_d": val_ks_d,
            "val_mass": val_mass,
            "val_crps": val_crps,
            "all_train_instances": num_train_instances is None
        }
    ]

    if fold != 0:
        return
    properties = ["skewness", "variance", "cv", "kurtosis", "iqr", "mean"]
    output_dir = "/home/anonymous/dist_explainability/paper_ijcai/figures/explainability"
    os.makedirs(output_dir, exist_ok=True)
    
    deleted_idx = norm_params.get('deleted_idx', [])
    try:
        deleted_idx_arr = np.asarray(deleted_idx).ravel()
        deleted_idx_set = set(int(x) for x in deleted_idx_arr.tolist())
    except Exception:
        deleted_idx_set = set()
    feature_names = [name for i, name in enumerate(feature_names_orig) if i not in deleted_idx_set]

    for prop in properties:
        try:
            property_func = get_property_function(prop, dist_name)
        except KeyError:
            print(f"Property function for {prop} not implemented for {dist_name}, skipping")
            continue

        print(f"Computing {prop} for scenario={scenario}, model={model_name}, dist={dist_name}, fold={fold}")
        
        fitted_params_for_property = []
        for i in range(len(val_runtimes_norm)):
            r_i = val_runtimes_norm[i]
            assert r_i.shape[0] == 100
            params = dist.fit(r_i, floc=0 if floc else None)
            fitted_params_for_property.append(params)
        fitted_params_for_property = np.array(fitted_params_for_property)
        
        y_property = property_func(fitted_params_for_property)
        if hasattr(y_property, 'numpy'):
            y_property = y_property.numpy()

        wrapped_model = PropertyWrapper(model, property_func)
        
        importance_start_time = time.time()
        importance_start_cpu = time.process_time()
        importance_result = compute_permutation_importance_local(wrapped_model, val_features, y_property, n_repeats=10)
        importance_end_cpu = time.process_time()
        importance_end_time = time.time()
        importance_wall_time = importance_end_time - importance_start_time
        importance_cpu_time = importance_end_cpu - importance_start_cpu
        print(f"  Permutation importance time: {importance_wall_time:.2f}s wall, {importance_cpu_time:.2f}s CPU")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importance_result.importances_mean,
            'importance_std': importance_result.importances_std
        }).sort_values('importance_mean', ascending=False)

        instances_suffix_local = f"_inst{num_train_instances}" if num_train_instances is not None else ""
        output_prefix = f"{scenario}_{model_name}_{dist_name}_{num_train_samples}{instances_suffix_local}_fold{fold}_{prop}"

        importance_df.to_csv(os.path.join(output_dir, f"{output_prefix}_permutation_importance.csv"), index=False)
        plot_permutation_importance_local(importance_result, feature_names, os.path.join(output_dir, f"{output_prefix}_permutation_importance.pdf"), property_name=prop.capitalize())

        perm_importance_data = {
            'importances': importance_result.importances,
            'importances_mean': importance_result.importances_mean,
            'importances_std': importance_result.importances_std,
            'feature_names': feature_names,
            'property_name': prop,
            'scenario': scenario,
            'model_name': model_name,
            'dist_name': dist_name,
            'fold': fold,
            'num_train_samples': num_train_samples,
            'num_train_instances': num_train_instances,
            'importance_wall_time': importance_wall_time,
            'importance_cpu_time': importance_cpu_time,
        }
        with lzma.open(os.path.join(output_dir, f"{output_prefix}_permutation_importance_data.pkl"), 'wb') as f:
            pickle.dump(perm_importance_data, f)

        top_features = importance_result.importances_mean.argsort()[::-1]
        ice_data_all = {}
        
        ice_start_time = time.time()
        ice_start_cpu = time.process_time()
        for feat_idx in top_features:
            grid_values, ice_curves, pdp_curve = compute_ice_curves_local(model, val_features, feat_idx, property_func, norm_params=norm_params)
            plot_ice_curves(grid_values, ice_curves, pdp_curve, feature_names[feat_idx], os.path.join(output_dir, f"{output_prefix}_ice_feature{feat_idx}.pdf"), property_name=prop.capitalize())
            
            ice_data_all[feat_idx] = {
                'grid_values': grid_values,
                'ice_curves': ice_curves,
                'pdp_curve': pdp_curve,
                'feature_name': feature_names[feat_idx],
                'feature_idx': feat_idx,
            }
        ice_end_cpu = time.process_time()
        ice_end_time = time.time()
        ice_wall_time = ice_end_time - ice_start_time
        ice_cpu_time = ice_end_cpu - ice_start_cpu
        print(f"  ICE curves time: {ice_wall_time:.2f}s wall, {ice_cpu_time:.2f}s CPU")
        
        ice_save_data = {
            'ice_data': ice_data_all,
            'property_name': prop,
            'scenario': scenario,
            'model_name': model_name,
            'dist_name': dist_name,
            'fold': fold,
            'num_train_samples': num_train_samples,
            'num_train_instances': num_train_instances,
            'feature_names': feature_names,
            'top_features': top_features.tolist(),
            'ice_wall_time': ice_wall_time,
            'ice_cpu_time': ice_cpu_time,
        }
        with lzma.open(os.path.join(output_dir, f"{output_prefix}_ice_data.pkl"), 'wb') as f:
            pickle.dump(ice_save_data, f)

    print(f"Analysis finished for scenario={scenario}, model={model_name}, dist={dist_name}, fold={fold}")

if __name__ == "__main__":
    

    executor = submitit.AutoExecutor("/logs", "slurm")
    executor.update_parameters(
        timeout_min=60 * 24 * 2,
        slurm_partition="CLUSTER",
        slurm_array_parallelism=1200,
        cpus_per_task=1,
        mem_gb=15.7 * 1,
        tasks_per_node=1,
        slurm_job_name="DIST",
    )

    os.makedirs("results/smac", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    dfs = []
    for scenario in get_sc_dict().keys():
        path = f"/home/anonymous/dist_explainability/results/results{scenario}_distnet_results.csv"

        df_temp = pd.read_csv(path)
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    for scenario in get_sc_dict().keys():
        df_scenario = df[df["scenario"] == scenario]
        max_inst = df_scenario["num_train_instances"].max()
        df.loc[df["num_train_instances"] == max_inst - 1, "num_train_instances"] = max_inst

    n_instances = None
    for model_name in ["xgb_dist"]:
        for n_samples in [100]:

            if model_name == "xgb_dist":
                if n_samples == 100:
                    n_cpus = 8
                elif n_samples >= 32:
                    n_cpus = 4
                else:
                    n_cpus = 1
            else:
                n_cpus = 1

            executor = submitit.AutoExecutor("/storage/work/anonymous/logs/dists", "slurm")
            executor.update_parameters(
                timeout_min=60 * 24 * 2,
                slurm_partition="Kathleen",
                slurm_array_parallelism=1200,
                cpus_per_task=n_cpus,
                mem_gb=15.7 * n_cpus,
                tasks_per_node=1,
                slurm_job_name="DIST",
                slurm_qos="medium",
            )

            with executor.batch():
                for scenario in get_sc_dict().keys():
                    for dist_name in [
                        "lognorm",
                    ]:

                        print("Running scenario %s" % scenario)
                        jobs = []

                        for fold in [0]:

                            if os.path.exists(
                                f"/home/anonymous/dist_explainability/results/models/{scenario}_{model_name}_{dist_name}_{n_samples}_fold{fold}.json"
                            ):
                                print(
                                    f"Skipping scenario {scenario} model {model_name} dist {dist_name} n_samples {n_samples} n_instances {n_instances} fold {fold} since model file exists"
                                )
                                continue
                            job = executor.submit(
                                run_fold,
                                scenario,
                                fold,
                                model_name,
                                dist_name,
                                n_samples,
                                n_instances,
                                n_cpus,
                            )
                            jobs.append(job)
