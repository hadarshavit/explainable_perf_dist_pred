import numpy as np
import pandas as pd
from typing import Callable

from sklearn.model_selection import KFold, train_test_split

try:
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.facade import AbstractFacade

    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Float,
        Integer,
        OrdinalHyperparameter,
    )
except Exception:
    ConfigurationSpace = None
    Categorical = None
    Float = None
    Integer = None
    OrdinalHyperparameter = None

import torch

from asf.epm.distnet import DistNet

from asf.predictors.utils.losses import lognorm_loss
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle

def tune_distnet(
    model: type[DistNet],
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    output_size: int = 2,
    *,
    validation_mode="cv",
    features_preprocessing: str | object = "default",
    categorical_features: list | None = None,
    numerical_features: list | None = None,
    groups: np.ndarray | None = None,
    cv: int = 5,
    timeout: int = 3600,
    runcount_limit: int = 100,
    output_dir: str = "./smac_output",
    seed: int = 0,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    smac_facade: AbstractFacade = HyperparameterOptimizationFacade,
    smac_scenario_kwargs: dict | None = None,
    smac_kwargs: dict | None = None,
    distnet_kwargs: dict | None = None,
) -> DistNet:
    assert SMAC_AVAILABLE, (
        "SMAC is not installed. Please install it to use this function."
    )

    smac_scenario_kwargs = smac_scenario_kwargs or {}
    smac_kwargs = smac_kwargs or {}
    distnet_kwargs = distnet_kwargs or {}

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(
            X, index=range(len(X)), columns=[f"f_{i}" for i in range(X.shape[1])]
        )
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y, index=range(len(y)))

    cs = model.get_configuration_space()

    scenario = Scenario(
        configspace=cs,
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=output_dir,
        seed=seed,
        **smac_scenario_kwargs,
    )

    loss_fn = metric or lognorm_loss

    def target_function_cv(config, seed):
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        fold_losses: list[float] = []
        for train_idx, test_idx in kfold.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            dn = model.get_from_configuration(
                input_size=X.shape[1],
                output_size=output_size,
                configuration=config,
                **distnet_kwargs,
            )
            dn = dn()

            dn.fit(X_train, y_train)

            with torch.no_grad():
                preds = dn.predict(X_test)
                y_true = torch.tensor(
                    y_test.to_numpy()
                    if isinstance(y_test, pd.Series)
                    else np.asarray(y_test),
                    dtype=torch.float32,
                )

                stds = []
                for col in range(preds.shape[1]):
                    y_true_col = preds[:, col]
                    col_std = np.std(y_true_col)
                    stds.append(col_std)

                loss_val = loss_fn(y_true, preds).item()
            fold_losses.append(loss_val)

        return float(np.mean(fold_losses)), {"stds": stds}

    def target_function_val_split(config, seed):
        X_train, X_test, y_train, y_test =train_test_split(X, y, random_state=seed, test_size=0.2)
        

        dn = model.get_from_configuration(
            input_size=X.shape[1],
            output_size=output_size,
            configuration=config,
            **distnet_kwargs,
        )
        dn = dn()

        dn.fit(X_train, y_train)

        with torch.no_grad():
            preds = dn.predict(X_test)
            y_true = torch.tensor(
                y_test.to_numpy()
                if isinstance(y_test, pd.Series)
                else np.asarray(y_test),
                dtype=torch.float32,
            )

            stds = []
            changing_col = False
            for col in range(preds.shape[1]):
                y_true_col = preds[:, col]
                if isinstance(y_true_col, torch.Tensor):
                    y_true_col = y_true_col.cpu().numpy()
                col_std = np.std(y_true_col)
                stds.append(col_std)
                if col_std > 0.01:
                    changing_col = True
            if not changing_col:
                raise ValueError("All columns have too small std, invalid configuration.")
            loss_val = loss_fn(y_true, preds).item()

        return loss_val, {"stds": stds}

    smac = smac_facade(scenario, target_function_cv if validation_mode == "cv" else target_function_val_split, **smac_kwargs)
    best_config = smac.optimize()

    best_dn = model.get_from_configuration(
        input_size=X.shape[1], output_size=output_size, configuration=best_config, **distnet_kwargs
    )

    return best_dn
