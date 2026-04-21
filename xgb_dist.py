from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
import pickle
import torch
import xgboost as xgb
import time
from asf.predictors.utils.losses import lognorm_loss
from asf.predictors.utils.mlp import ExpActivation

try:
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        Constant,
        EqualsCondition,
        Float,
        Integer,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

class XGBDistNet:
    PREFIX = "xgb_distnet"

    def __init__(
        self,
        loss_function=lognorm_loss,
        n_loss_params: int = 2,
        batch_size: int | None = 1000,
        output_activation=ExpActivation(),
        stabilization: str = "MAD",
        use_start_values: bool = True,
        early_stopping_rounds: int | None = None,
        early_stopping_tolerance: float = 0.0,
        device: str = "cpu",
        **kwargs,
    ):
        self.loss_function = loss_function
        self.n_loss_params = n_loss_params
        self.batch_size = batch_size
        self.output_activation = output_activation
        self.stabilization = stabilization
        self.use_start_values = use_start_values
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = early_stopping_tolerance

        self.device = device
        self._is_gpu_device = str(device).lower().startswith("cuda") or str(device).lower().startswith("gpu")
        self.kwargs = kwargs
        self.kwargs["device"] = device
        if "callbacks" not in self.kwargs:
            self.kwargs["callbacks"] = []
        if self.early_stopping_rounds is not None:
            self.kwargs["callbacks"].append(
                xgb.callback.EarlyStopping(
                    rounds=self.early_stopping_rounds,
                    save_best=True,
                    min_delta=self.early_stopping_tolerance,
                )
            )
        self.kwargs[ "disable_default_eval_metric"] = True
        if self._is_gpu_device:
            self.kwargs.setdefault("tree_method", "hist")
            self.kwargs.setdefault("predictor", "gpu_predictor")
            self.kwargs.setdefault("single_precision_histogram", True)

        self._objective_target_tensor: torch.Tensor | None = None
        self._objective_target_ptr: int | None = None

    def stabilize_derivative(
        self, input_der: torch.Tensor, type: str = "MAD"
    ) -> torch.Tensor:
        if type == "MAD":
            input_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))
            div = torch.median(torch.abs(input_der - torch.nanmedian(input_der)))
            div = torch.where(div < torch.tensor(1e-04), torch.tensor(1e-04), div)
            stab_der = input_der / div
        elif type == "L2":
            input_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))
            div = torch.sqrt(torch.mean(input_der.pow(2)))
            div = torch.where(div < torch.tensor(1e-04), torch.tensor(1e-04), div)
            div = torch.where(div > torch.tensor(10000.0), torch.tensor(10000.0), div)
            stab_der = input_der / div
        else:
            stab_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))
        return stab_der

    def _clear_objective_cache(self) -> None:
        self._objective_target_tensor = None
        self._objective_target_ptr = None

    def _get_cached_target_tensor(self, data: np.ndarray | xgb.DMatrix | Any) -> torch.Tensor:
        if hasattr(data, "get_label"):
            labels = data.get_label()
        else:
            labels = np.asarray(data)
        ptr = int(labels.__array_interface__["data"][0])
        if (
            self._objective_target_tensor is None
            or self._objective_target_ptr != ptr
            or self._objective_target_tensor.shape[0] != labels.size
        ):
            target = torch.from_numpy(labels.reshape(-1, 1)).to(
                self.device, dtype=torch.float32
            )
            self._objective_target_tensor = target
            self._objective_target_ptr = ptr
        return self._objective_target_tensor

    def objective(self, data, preds):
        start = time.time()
        preds = np.asarray(preds, dtype=np.float32)
        if self.n_loss_params == 1:
            preds = preds.reshape(-1, 1)

        target = self._get_cached_target_tensor(data)

        if self.batch_size is None:
            raw = torch.from_numpy(preds).requires_grad_(True).float().to(self.device)
            activated = self.output_activation(raw)
            activated = torch.clamp(activated, min=1e-12)

            loss = self.loss_function(target, activated)

            grad_raw = torch.autograd.grad(loss, inputs=raw, create_graph=True)[0]

            if grad_raw.dim() == 1:
                hess_raw = torch.autograd.grad(
                    grad_raw.sum(), inputs=raw, retain_graph=True
                )[0]
            else:
                h_cols = []
                for d in range(grad_raw.shape[1]):
                    gcol_sum = grad_raw[:, d].sum()
                    h_col_full = torch.autograd.grad(
                        gcol_sum, inputs=raw, retain_graph=True
                    )[0]
                    h_cols.append(h_col_full[:, d])
                hess_raw = torch.stack(h_cols, dim=1)

            if self.stabilization != "None":
                if grad_raw.dim() == 1:
                    grad_raw = self.stabilize_derivative(
                        grad_raw, type=self.stabilization
                    )
                    hess_raw = self.stabilize_derivative(
                        hess_raw, type=self.stabilization
                    )
                else:
                    for d in range(grad_raw.shape[1]):
                        grad_raw[:, d] = self.stabilize_derivative(
                            grad_raw[:, d], type=self.stabilization
                        )
                        hess_raw[:, d] = self.stabilize_derivative(
                            hess_raw[:, d], type=self.stabilization
                        )

            grad = grad_raw.detach().cpu().numpy()
            hess = hess_raw.detach().cpu().numpy()

            return grad, hess
        else:
            n_samples = target.shape[0]
            preds_tensor = torch.from_numpy(preds).to(self.device)
            grad = torch.zeros_like(preds_tensor)
            hess = torch.zeros_like(preds_tensor)

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)

                batch_raw = (
                    torch.from_numpy(preds[start_idx:end_idx])
                    .requires_grad_(True)
                    .float()
                    .to(self.device)
                )
                batch_activated = self.output_activation(batch_raw)
                batch_activated = torch.clamp(batch_activated, min=1e-12)
                batch_target = target[start_idx:end_idx]
                batch_loss = self.loss_function(batch_target, batch_activated)

                batch_grad_raw = torch.autograd.grad(
                    batch_loss, inputs=batch_raw, create_graph=True
                )[0]

                if batch_grad_raw.dim() == 1:
                    batch_hess_raw = torch.autograd.grad(
                        batch_grad_raw.sum(), inputs=batch_raw, retain_graph=True
                    )[0]
                else:
                    h_cols = []
                    for d in range(batch_grad_raw.shape[1]):
                        gcol_sum = batch_grad_raw[:, d].sum()
                        h_col_full = torch.autograd.grad(
                            gcol_sum, inputs=batch_raw, retain_graph=True
                        )[0]
                        h_cols.append(h_col_full[:, d])
                    batch_hess_raw = torch.stack(h_cols, dim=1)

                grad[start_idx:end_idx] = batch_grad_raw.detach()
                hess[start_idx:end_idx] = batch_hess_raw.detach()

            if self.stabilization != "None":
                if grad.dim() == 1:
                    grad = self.stabilize_derivative(grad, type=self.stabilization)
                    hess = self.stabilize_derivative(hess, type=self.stabilization)
                else:
                    for d in range(grad.shape[1]):
                        grad[:, d] = self.stabilize_derivative(
                            grad[:, d], type=self.stabilization
                        )
                        hess[:, d] = self.stabilize_derivative(
                            hess[:, d], type=self.stabilization
                        )

            grad = grad.cpu().numpy()
            hess = hess.cpu().numpy()

            end = time.time()

            return grad, hess

    def calculate_start_values(
        self, target: np.ndarray, max_iter: int = 50
    ) -> np.ndarray:
        from torch.optim import LBFGS
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        target_tensor = (
            torch.tensor(target, dtype=torch.float32).reshape(-1, 1).to(self.device)
        )

        params = [
            torch.tensor(0.5, requires_grad=True, device=self.device) for _ in range(self.n_loss_params)
        ]

        optimizer = LBFGS(
            params,
            lr=0.1,
            max_iter=np.min([int(max_iter / 4), 20]),
            line_search_fn="strong_wolfe",
        )
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        def closure():
            optimizer.zero_grad()

            params_stack = torch.stack(params)
            nan_inf_idx = torch.isnan(params_stack) | torch.isinf(params_stack)
            params_clean = torch.where(nan_inf_idx, torch.tensor(0.5), params_stack)

            params_activated = [
                self.output_activation(params_clean[i].reshape(-1, 1))
                for i in range(self.n_loss_params)
            ]

            if self.n_loss_params == 1:
                params_tensor = params_activated[0]
            else:
                params_tensor = torch.cat(params_activated, dim=1)

            params_replicated = params_tensor.repeat(len(target_tensor), 1)

            loss = self.loss_function(target_tensor, params_replicated)
            loss.backward()
            return loss

        loss_vals = []
        for epoch in range(max_iter):
            loss = optimizer.step(closure)
            lr_scheduler.step(loss)
            loss_vals.append(loss.item())

        start_values = np.array(
            [params[i].detach().cpu().numpy() for i in range(self.n_loss_params)]
        )

        start_values = np.nan_to_num(start_values, nan=0.5, posinf=0.5, neginf=0.5)

        return start_values

    def fit(
        self,
        X: pd.DataFrame | pd.Series | list,
        y: pd.Series | list,
    ):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X = np.concatenate([[x for i in range(y.shape[1])] for x in X]).astype(
            np.float32, copy=False
        )
        y = y.flatten().astype(np.float32, copy=False)

        self._clear_objective_cache()

        if self.use_start_values:
            start_values = self.calculate_start_values(y, max_iter=50)
            print(f"Calculated start values (pre-activation): {start_values}")
        else:
            start_values = np.zeros(self.n_loss_params)
            print("Using zero initialization (start values disabled)")
        start_values = start_values.astype(np.float32, copy=False)

        if self.n_loss_params == 1:
            base_margin = np.full(len(y), start_values[0], dtype=np.float32)
        else:
            base_margin = np.tile(start_values, (len(y), 1)).astype(
                np.float32, copy=False
            )

        def _eval_metric(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> float:
            if self.n_loss_params == 1:
                y_pred_np = y_pred_np.reshape(-1, 1)
            else:
                y_pred_np = y_pred_np.reshape(-1, self.n_loss_params)
            y_true_t = torch.from_numpy(y_true_np.astype(np.float32)).reshape(-1, 1)
            y_pred_t = torch.from_numpy(y_pred_np.astype(np.float32))
            activated = self.output_activation(y_pred_t)

            return float(self.loss_function(y_true_t, activated))

        self.model = xgb.XGBRegressor(
            objective=self.objective,
            eval_metric=_eval_metric,
            num_target=self.n_loss_params,
            **self.kwargs,
        )

        self.start_values = start_values

        self.model.fit(
            X,
            y,
            eval_set=[(X, y)],
            verbose=False,
            base_margin=base_margin,
        )
        self._clear_objective_cache()

    def predict(self, X: pd.DataFrame | pd.Series | list) -> torch.Tensor:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        n_samples = X.shape[0]
        if self.n_loss_params == 1:
            base_margin_pred = np.full(n_samples, self.start_values[0], dtype=np.float32)
        else:
            base_margin_pred = np.tile(self.start_values, (n_samples, 1)).astype(
                np.float32, copy=False
            )

        predictions = self.model.predict(X, base_margin=base_margin_pred)
        preds_tensor = self.output_activation(torch.from_numpy(predictions).to(self.device))
        preds_tensor = torch.clamp(preds_tensor, min=1e-12)
        predictions = preds_tensor.cpu().numpy()

        if self.n_loss_params == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def save(self, path: str):
        self.model.save_model(path)

        def _to_serializable(obj):
            try:
                pickle.dumps(obj)
                return {'__pickled__': True, 'value': obj}
            except Exception:
                module = getattr(obj, '__module__', None)
                name = getattr(obj, '__name__', None)
                if module and name:
                    return {'__pickled__': False, '__kind__': 'callable', 'module': module, 'name': name}
                cls = obj.__class__
                return {
                    '__pickled__': False,
                    '__kind__': 'instance',
                    'class_module': cls.__module__,
                    'class_name': cls.__name__,
                    'repr': repr(obj),
                }

        metadata = {
            'start_values': self.start_values,
            'n_loss_params': self.n_loss_params,
            'output_activation': _to_serializable(self.output_activation),
            'loss_function': _to_serializable(self.loss_function),
            'stabilization': self.stabilization,
            'device': self.device,
            'batch_size': self.batch_size,
            'use_start_values': self.use_start_values,
            'early_stopping_rounds': self.early_stopping_rounds,
            'early_stopping_tolerance': self.early_stopping_tolerance,
            'kwargs': {},
        }

        for k, v in (self.kwargs or {}).items():
            try:
                pickle.dumps(v)
                metadata['kwargs'][k] = {'__pickled__': True, 'value': v}
            except Exception:
                module = getattr(v, '__module__', None)
                name = getattr(v, '__name__', None)
                if module and name:
                    metadata['kwargs'][k] = {'__pickled__': False, '__kind__': 'callable', 'module': module, 'name': name}
                else:
                    cls = v.__class__
                    metadata['kwargs'][k] = {'__pickled__': False, '__kind__': 'instance', 'class_module': cls.__module__, 'class_name': cls.__name__, 'repr': repr(v)}

        metadata_path = path + '.metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, path: str):
        import importlib

        def _from_serializable(entry, default=None):
            if not isinstance(entry, dict) or '__pickled__' not in entry:
                return entry
            if entry.get('__pickled__'):
                return entry.get('value')
            kind = entry.get('__kind__')
            try:
                if kind == 'callable':
                    mod = importlib.import_module(entry['module'])
                    return getattr(mod, entry['name'])
                elif kind == 'instance':
                    mod = importlib.import_module(entry['class_module'])
                    cls = getattr(mod, entry['class_name'])
                    return cls()
            except Exception:
                return default
            return default

        metadata_path = path + '.metadata.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        output_activation = _from_serializable(metadata.get('output_activation'), default=None)
        loss_function = _from_serializable(metadata.get('loss_function'), default=None)

        kwargs = {}
        for k, v in metadata.get('kwargs', {}).items():
            kwargs[k] = _from_serializable(v, default=None)

        if output_activation is None:
            output_activation = ExpActivation()
        if loss_function is None:
            from asf.predictors.utils.losses import lognorm_loss as _default_loss

            loss_function = _default_loss

        instance = cls(
            loss_function=loss_function,
            n_loss_params=metadata['n_loss_params'],
            batch_size=metadata.get('batch_size'),
            output_activation=output_activation,
            stabilization=metadata['stabilization'],
            use_start_values=metadata.get('use_start_values', True),
            early_stopping_rounds=metadata.get('early_stopping_rounds'),
            early_stopping_tolerance=metadata.get('early_stopping_tolerance', 0.0),
            device=metadata['device'],
            **(kwargs or {}),
        )

        instance.model = xgb.XGBRegressor()
        instance.model.load_model(path)

        instance.start_values = metadata['start_values']

        return instance
    
    def load_old_format(self, path: str):
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        
        metadata_path = path + '.metadata.pkl'
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.start_values = metadata['start_values']
        except FileNotFoundError:
            self.start_values = np.zeros(self.n_loss_params, dtype=np.float32)

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
    ) -> ConfigurationSpace:
        if cs is None:
            cs = ConfigurationSpace(name="XGBoostRegressor")

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBDistNet.PREFIX}"
        else:
            prefix = XGBDistNet.PREFIX

        booster = Constant(f"{prefix}:booster", "gbtree")
        n_estimators = Constant(f"{prefix}:n_estimators", 2000)
        max_depth = Integer(
            f"{prefix}:max_depth",
            (3, 15),
            log=False,
        )
        min_child_weight = Integer(
            f"{prefix}:min_child_weight",
            (1, 20),
            log=True,
        )
        colsample_bytree = Float(
            f"{prefix}:colsample_bytree",
            (0.3, 1.0),
            log=False,
        )
        colsample_bylevel = Float(
            f"{prefix}:colsample_bylevel",
            (0.3, 1.0),
            log=False,
        )
        lambda_param = Float(
            f"{prefix}:lambda",
            (0.001, 10),
            log=True,
        )
        alpha = Float(
            f"{prefix}:alpha",
            (0.001, 10),
            log=True,
        )
        learning_rate = Float(
            f"{prefix}:learning_rate",
            (0.001, 0.3),
            log=True,
        )
        multi_strategy = Categorical(
            f"{prefix}:multi_strategy", ["one_output_per_tree"]
        )
        stabilization = Categorical(
            f"{prefix}:stabilization", ["MAD", "None"]
        )
        use_start_values = Categorical(
            f"{prefix}:use_start_values", [True, False], default=True
        )

        params = [
            booster,
            n_estimators,
            max_depth,
            min_child_weight,
            colsample_bytree,
            colsample_bylevel,
            lambda_param,
            alpha,
            learning_rate,
            multi_strategy,
            stabilization,
            use_start_values,
        ]
        if parent_param is not None:
            conditions = [
                EqualsCondition(
                    child=param,
                    parent=parent_param,
                    value=parent_value,
                )
                for param in params
            ]
        else:
            conditions = []

        cs.add(params + conditions)

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: dict[str, Any],
        pre_prefix: str = "",
        input_size: int = None,
        **kwargs,
    ) -> Callable[..., "XGBDistNet"]:
        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBDistNet.PREFIX}"
        else:
            prefix = XGBDistNet.PREFIX

        xgb_params = {
            "booster": configuration[f"{prefix}:booster"],
            "n_estimators": configuration[f"{prefix}:n_estimators"],
            "max_depth": configuration[f"{prefix}:max_depth"],
            "min_child_weight": configuration[f"{prefix}:min_child_weight"],
            "colsample_bytree": configuration[f"{prefix}:colsample_bytree"],
            "colsample_bylevel": configuration[f"{prefix}:colsample_bylevel"],
            "lambda": configuration[f"{prefix}:lambda"],
            "alpha": configuration[f"{prefix}:alpha"],
            "learning_rate": configuration[f"{prefix}:learning_rate"],
            "multi_strategy": configuration[f"{prefix}:multi_strategy"],
            "stabilization": configuration[f"{prefix}:stabilization"],
            "use_start_values": configuration[f"{prefix}:use_start_values"],
            **kwargs,
        }

        return partial(XGBDistNet, **xgb_params)