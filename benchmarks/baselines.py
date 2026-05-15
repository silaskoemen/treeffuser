from __future__ import annotations

import math
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
from jaxtyping import Float
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SampleBaseline:
    """Small adapter protocol for external probabilistic baselines."""

    def fit(
        self,
        X: Float[ndarray, "train x_dim"],
        y: Float[ndarray, "train y_dim"],
    ) -> SampleBaseline:
        raise NotImplementedError

    def sample(
        self,
        X: Float[ndarray, "batch x_dim"],
        n_samples: int = 200,
        seed: int | None = None,
        **kwargs,
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        raise NotImplementedError


def _require(package: str, install_hint: str):
    try:
        return __import__(package)
    except ImportError as exc:
        raise ImportError(f"Benchmark baseline requires optional dependency {package!r}. {install_hint}") from exc


class ScaledRegressorMixin:
    def _fit_scalers(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        X_scaled = self._x_scaler.fit_transform(X)
        y_scaled = self._y_scaler.fit_transform(_ensure_2d_y(y))
        return X_scaled, y_scaled

    def _transform_x(self, X: ndarray) -> ndarray:
        return self._x_scaler.transform(X)

    def _inverse_y(self, y: ndarray) -> ndarray:
        shape = y.shape
        y_2d = y.reshape(-1, shape[-1])
        return self._y_scaler.inverse_transform(y_2d).reshape(shape)


def _ensure_2d_y(y: ndarray) -> ndarray:
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


class NGBoostGaussianBaseline(ScaledRegressorMixin, SampleBaseline):
    def __init__(
        self,
        n_estimators: int = 5000,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 20,
        seed: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.model = None

    def fit(self, X: ndarray, y: ndarray) -> NGBoostGaussianBaseline:
        ngboost = _require("ngboost", "Install the bench extras with pixi before running.")
        y = _ensure_2d_y(y)
        X_scaled, y_scaled = self._fit_scalers(X, y)
        if y_scaled.shape[1] != 1:
            raise ValueError("NGBoostGaussianBaseline currently supports one-dimensional y.")
        minibatch_frac = min(50_000, X_scaled.shape[0]) / X_scaled.shape[0]
        validation_fraction = min(int(0.1 * X_scaled.shape[0]), 20_000) / X_scaled.shape[0]
        self.model = ngboost.NGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            minibatch_frac=minibatch_frac,
            validation_fraction=validation_fraction,
            verbose=False,
            random_state=self.seed,
        )
        self.model.fit(X_scaled, y_scaled[:, 0])
        return self

    def sample(self, X: ndarray, n_samples: int = 200, seed: int | None = None, **kwargs) -> ndarray:
        del kwargs
        X_scaled = self._transform_x(X)
        np.random.seed(seed)
        samples = np.asarray(self.model.pred_dist(X_scaled).sample(n_samples)).reshape(n_samples, -1, 1)
        return self._inverse_y(samples)


class IBUGXGBoostBaseline(SampleBaseline):
    def __init__(
        self,
        k: int = 100,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        leaf_sample_trees: int = 64,
        seed: int | None = None,
    ) -> None:
        self.k = k
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.leaf_sample_trees = leaf_sample_trees
        self.seed = seed
        self.model = None
        self.gbrt_model = None
        self._train_leaves = None
        self._train_y = None

    def fit(self, X: ndarray, y: ndarray) -> IBUGXGBoostBaseline:
        xgboost = _require("xgboost", "Install the bench extras with pixi before running.")
        y = _ensure_2d_y(y)
        if y.shape[1] != 1:
            raise ValueError("IBUGXGBoostBaseline currently supports one-dimensional y.")
        y_1d = y[:, 0]
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_1d,
            test_size=0.1,
            random_state=self.seed,
        )
        gbrt_model = xgboost.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.seed,
            objective="reg:squarederror",
        ).fit(X_train, y_train)
        self.gbrt_model = gbrt_model
        try:
            from ibug import IBUGWrapper  # noqa: PLC0415
        except ImportError:
            self.model = None
            self._train_leaves = gbrt_model.apply(X_train)
            self._train_y = y_train
        else:
            self.model = IBUGWrapper(k=self.k).fit(
                gbrt_model,
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
            )
        return self

    def sample(self, X: ndarray, n_samples: int = 200, seed: int | None = None, **kwargs) -> ndarray:
        del kwargs
        rng = np.random.default_rng(seed)
        if self.model is not None:
            location, scale = self.model.pred_dist(X)
            scale = np.maximum(scale, 1e-12)
            return rng.normal(location, scale, size=(n_samples, X.shape[0]))[:, :, None]

        test_leaves = self.gbrt_model.apply(X)
        samples = np.empty((n_samples, X.shape[0]), dtype=np.float64)
        for i, leaves in enumerate(test_leaves):
            affinity = np.mean(
                self._train_leaves[:, : self.leaf_sample_trees] == leaves[None, : self.leaf_sample_trees],
                axis=1,
            )
            k = min(self.k, affinity.shape[0])
            neighbor_idx = np.argpartition(affinity, -k)[-k:]
            draw_idx = rng.choice(neighbor_idx, size=n_samples, replace=True)
            samples[:, i] = self._train_y[draw_idx]
        return samples[:, :, None]


class DistributionalRandomForestBaseline(SampleBaseline):
    def __init__(
        self,
        min_node_size: int = 10,
        num_trees: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.min_node_size = min_node_size
        self.num_trees = num_trees
        self.seed = seed
        self.model = None

    def fit(self, X: ndarray, y: ndarray) -> DistributionalRandomForestBaseline:
        drf_pkg = _require(
            "drf",
            "DRF is not a normal pixi dependency; install the R-backed drf package as documented in testbed.",
        )
        del self.seed
        self.model = drf_pkg.drf(
            min_node_size=self.min_node_size,
            num_trees=self.num_trees,
            splitting_rule="FourierMMD",
        )
        self.model.fit(X, _ensure_2d_y(y))
        return self

    def sample(self, X: ndarray, n_samples: int = 200, seed: int | None = None, **kwargs) -> ndarray:
        del kwargs
        np.random.seed(seed)
        out = self.model.predict(newdata=X, functional="sample", n=n_samples).sample
        return np.transpose(out, (2, 0, 1))


class LightGBMQuantileBaseline(SampleBaseline):
    def __init__(
        self,
        quantile_count: int = 99,
        n_estimators: int = 3000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        n_jobs: int = -1,
        early_stopping_rounds: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.quantile_count = quantile_count
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.models: list[Any] = []
        self.n_estimators_true: list[int] = []
        self.quantiles = np.linspace(1.0 / (quantile_count + 1), quantile_count / (quantile_count + 1), quantile_count)

    def fit(self, X: ndarray, y: ndarray) -> LightGBMQuantileBaseline:
        y = _ensure_2d_y(y)
        if y.shape[1] != 1:
            raise ValueError("LightGBMQuantileBaseline currently supports one-dimensional y.")
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y[:, 0],
            test_size=0.1,
            random_state=self.seed,
        )
        self.models = []
        self.n_estimators_true = []
        for alpha in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=float(alpha),
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                n_jobs=self.n_jobs,
                random_state=self.seed,
                verbose=-1,
            )
            fit_kwargs: dict[str, Any] = {}
            if self.early_stopping_rounds is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            model.fit(X_train, y_train, **fit_kwargs)
            self.models.append(model)
            self.n_estimators_true.append(getattr(model, "best_iteration_", None) or self.n_estimators)
        return self

    def sample(self, X: ndarray, n_samples: int = 200, seed: int | None = None, **kwargs) -> ndarray:
        del kwargs
        rng = np.random.default_rng(seed)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            quantile_preds = np.stack([model.predict(X) for model in self.models], axis=0)
        quantile_preds = np.sort(quantile_preds, axis=0)
        uniforms = rng.uniform(size=(n_samples, X.shape[0]))
        samples = np.empty((n_samples, X.shape[0]), dtype=np.float64)
        for i in range(X.shape[0]):
            samples[:, i] = np.interp(uniforms[:, i], self.quantiles, quantile_preds[:, i])
        return samples[:, :, None]


class DeepEnsembleBaseline(ScaledRegressorMixin, SampleBaseline):
    """Lean PyTorch deep ensemble baseline.

    Deep ensembles are a standard, stable neural UQ comparator with tractable CPU runs.
    """

    def __init__(
        self,
        n_ensembles: int = 5,
        hidden_size: int = 100,
        n_layers: int = 2,
        max_epochs: int = 300,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        patience: int = 20,
        seed: int | None = None,
    ) -> None:
        self.n_ensembles = n_ensembles
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.models = []

    def fit(self, X: ndarray, y: ndarray) -> DeepEnsembleBaseline:
        torch = _require("torch", "Install the bench extras with pixi before running.")
        y = _ensure_2d_y(y)
        X_scaled, y_scaled = self._fit_scalers(X, y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled,
            y_scaled,
            test_size=0.1,
            random_state=self.seed,
        )
        X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        self.models = []
        for ensemble_idx in range(self.n_ensembles):
            if self.seed is not None:
                torch.manual_seed(self.seed + ensemble_idx)
            model = _TorchMeanVarianceMLP(
                x_dim=X_scaled.shape[1],
                y_dim=y_scaled.shape[1],
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed((self.seed or 0) + ensemble_idx),
            )
            best_state = None
            best_loss = math.inf
            stale_epochs = 0
            for _ in range(self.max_epochs):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    mean, var = model(xb)
                    loss = (0.5 * (torch.log(var) + (yb - mean) ** 2 / var)).mean()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    mean_val, var_val = model(X_val_t)
                    val_loss = (0.5 * (torch.log(var_val) + (y_val_t - mean_val) ** 2 / var_val)).mean().item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
        return self

    def sample(self, X: ndarray, n_samples: int = 200, seed: int | None = None, **kwargs) -> ndarray:
        del kwargs
        torch = _require("torch", "Install the bench extras with pixi before running.")
        rng = np.random.default_rng(seed)
        X_t = torch.as_tensor(self._transform_x(X), dtype=torch.float32)
        per_model_samples = []
        with torch.no_grad():
            for model in self.models:
                mean, var = model(X_t)
                generator = torch.Generator().manual_seed(int(rng.integers(2**31)))
                noise = torch.randn((n_samples, *mean.shape), generator=generator)
                per_model_samples.append((mean[None, :, :] + noise * torch.sqrt(var)[None, :, :]).numpy())
        stacked = np.concatenate(per_model_samples, axis=0)
        draw_idx = rng.choice(stacked.shape[0], size=n_samples, replace=False)
        return self._inverse_y(stacked[draw_idx])


class CARDRegressionBaseline(ScaledRegressorMixin, SampleBaseline):
    """Compact PyTorch implementation of CARD-style probabilistic regression.

    CARD trains a deterministic conditional mean model and then a conditional
    diffusion model for y. This adapter keeps that two-stage structure while
    avoiding the old testbed's Lightning dependency stack.
    """

    def __init__(
        self,
        hidden_size: int = 100,
        n_layers: int = 2,
        max_epochs: int = 150,
        diffusion_epochs: int | None = None,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        patience: int = 15,
        n_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.01,
        sample_batch_size: int = 4096,
        seed: int | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.diffusion_epochs = diffusion_epochs or max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.dropout = dropout
        self.sample_batch_size = sample_batch_size
        self.seed = seed
        self.cond_model = None
        self.diff_model = None
        self._betas = None
        self._alphas = None
        self._alpha_bars = None

    def fit(self, X: ndarray, y: ndarray) -> CARDRegressionBaseline:
        torch = _require("torch", "Install the bench extras with pixi before running.")
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        y = _ensure_2d_y(y)
        X_scaled, y_scaled = self._fit_scalers(X, y)
        X_scaled = X_scaled.astype(np.float32)
        y_scaled = y_scaled.astype(np.float32)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled,
            y_scaled,
            test_size=0.1,
            random_state=self.seed,
        )
        X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

        self.cond_model = _TorchPlainMLP(
            in_dim=X_scaled.shape[1],
            out_dim=y_scaled.shape[1],
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )
        self._fit_conditional_model(torch, X_train_t, y_train_t, X_val_t, y_val_t)

        self._betas, self._alphas, self._alpha_bars = _card_diffusion_schedule(
            torch=torch,
            n_steps=self.n_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )
        with torch.no_grad():
            y_hat_train = self.cond_model(X_train_t).detach()
            y_hat_val = self.cond_model(X_val_t).detach()
        self.diff_model = _TorchCARDDenoiser(
            x_dim=X_scaled.shape[1],
            y_dim=y_scaled.shape[1],
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )
        self._fit_diffusion_model(torch, X_train_t, y_train_t, y_hat_train, X_val_t, y_val_t, y_hat_val)
        return self

    def sample(self, X: ndarray, n_samples: int = 200, seed: int | None = None, **kwargs) -> ndarray:
        del kwargs
        torch = _require("torch", "Install the bench extras with pixi before running.")
        if self.cond_model is None or self.diff_model is None:
            raise ValueError("CARDRegressionBaseline must be fit before sampling.")

        X_t = torch.as_tensor(self._transform_x(X).astype(np.float32), dtype=torch.float32)
        repeated_X = X_t.repeat((n_samples, 1))
        generator = torch.Generator().manual_seed(seed or 0)
        sample_chunks = []
        self.cond_model.eval()
        self.diff_model.eval()
        with torch.no_grad():
            for start in range(0, repeated_X.shape[0], self.sample_batch_size):
                x_batch = repeated_X[start : start + self.sample_batch_size]
                y_batch = torch.randn(
                    (x_batch.shape[0], self._y_scaler.n_features_in_),
                    generator=generator,
                    dtype=torch.float32,
                )
                y_hat = self.cond_model(x_batch)
                for step in range(self.n_steps - 1, -1, -1):
                    t = torch.full((x_batch.shape[0], 1), step / max(self.n_steps - 1, 1))
                    eps = self.diff_model(x_batch, y_batch, y_hat, t)
                    beta = self._betas[step]
                    alpha = self._alphas[step]
                    alpha_bar = self._alpha_bars[step]
                    mean = (y_batch - beta * eps / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
                    if step > 0:
                        alpha_bar_prev = self._alpha_bars[step - 1]
                        variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                        noise = torch.randn(y_batch.shape, generator=generator, dtype=torch.float32)
                        y_batch = mean + torch.sqrt(variance) * noise
                    else:
                        y_batch = mean
                sample_chunks.append(y_batch.numpy())
        samples = np.concatenate(sample_chunks, axis=0).reshape(n_samples, X.shape[0], -1)
        return self._inverse_y(samples)

    def _fit_conditional_model(self, torch, X_train, y_train, X_val, y_val) -> None:
        optimizer = torch.optim.Adam(self.cond_model.parameters(), lr=self.learning_rate)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed or 0),
        )
        self._fit_torch_model(
            torch=torch,
            model=self.cond_model,
            optimizer=optimizer,
            loader=loader,
            max_epochs=self.max_epochs,
            validation_loss=lambda: torch.nn.functional.mse_loss(self.cond_model(X_val), y_val),
            batch_loss=lambda xb, yb: torch.nn.functional.mse_loss(self.cond_model(xb), yb),
        )

    def _fit_diffusion_model(self, torch, X_train, y_train, y_hat_train, X_val, y_val, y_hat_val) -> None:
        optimizer = torch.optim.Adam(self.diff_model.parameters(), lr=self.learning_rate)
        dataset = torch.utils.data.TensorDataset(X_train, y_train, y_hat_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed((self.seed or 0) + 1),
        )
        train_generator = torch.Generator().manual_seed((self.seed or 0) + 2)
        val_generator = torch.Generator().manual_seed((self.seed or 0) + 3)
        self._fit_torch_model(
            torch=torch,
            model=self.diff_model,
            optimizer=optimizer,
            loader=loader,
            max_epochs=self.diffusion_epochs,
            validation_loss=lambda: self._diffusion_loss(torch, X_val, y_val, y_hat_val, val_generator),
            batch_loss=lambda xb, yb, yhat: self._diffusion_loss(torch, xb, yb, yhat, train_generator),
        )

    def _fit_torch_model(
        self,
        torch,
        model,
        optimizer,
        loader,
        max_epochs: int,
        validation_loss,
        batch_loss,
    ) -> None:
        best_state = None
        best_loss = math.inf
        stale_epochs = 0
        for _ in range(max_epochs):
            model.train()
            for batch in loader:
                optimizer.zero_grad()
                loss = batch_loss(*batch)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = validation_loss().item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

    def _diffusion_loss(self, torch, X, y, y_hat, generator) -> Any:
        t_idx = torch.randint(self.n_steps, (X.shape[0],), generator=generator)
        alpha_bar = self._alpha_bars[t_idx].reshape(-1, 1)
        eps = torch.randn(y.shape, generator=generator, dtype=torch.float32)
        y_t = torch.sqrt(alpha_bar) * y + torch.sqrt(1.0 - alpha_bar) * eps
        t = t_idx.reshape(-1, 1).float() / max(self.n_steps - 1, 1)
        eps_pred = self.diff_model(X, y_t, y_hat, t)
        return torch.nn.functional.mse_loss(eps_pred, eps)


class _TorchMeanVarianceMLP:
    def __new__(cls, x_dim: int, y_dim: int, hidden_size: int, n_layers: int):
        torch = _require("torch", "Install the bench extras with pixi before running.")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers = []
                in_dim = x_dim
                for _ in range(n_layers):
                    layers.append(torch.nn.Linear(in_dim, hidden_size))
                    layers.append(torch.nn.ReLU())
                    in_dim = hidden_size
                layers.append(torch.nn.Linear(in_dim, 2 * y_dim))
                self.net = torch.nn.Sequential(*layers)

            def forward(self, x):
                raw = self.net(x)
                mean, raw_var = raw[:, :y_dim], raw[:, y_dim:]
                return mean, torch.nn.functional.softplus(raw_var) + 1e-6

        return Model()


class _TorchPlainMLP:
    def __new__(
        cls,
        in_dim: int,
        out_dim: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
    ):
        torch = _require("torch", "Install the bench extras with pixi before running.")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers = []
                current_dim = in_dim
                for _ in range(n_layers):
                    layers.append(torch.nn.Linear(current_dim, hidden_size))
                    layers.append(torch.nn.ReLU())
                    if dropout > 0.0:
                        layers.append(torch.nn.Dropout(dropout))
                    current_dim = hidden_size
                layers.append(torch.nn.Linear(current_dim, out_dim))
                self.net = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        return Model()


class _TorchCARDDenoiser:
    def __new__(
        cls,
        x_dim: int,
        y_dim: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
    ):
        torch = _require("torch", "Install the bench extras with pixi before running.")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = _TorchPlainMLP(
                    in_dim=x_dim + 2 * y_dim + 1,
                    out_dim=y_dim,
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                    dropout=dropout,
                )

            def forward(self, x, y_t, y_hat, t):
                return self.net(torch.cat([x, y_t, y_hat, t], dim=1))

        return Model()


def _card_diffusion_schedule(torch, n_steps: int, beta_start: float, beta_end: float):
    betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


class CatBoostUncertaintyBaseline(SampleBaseline):
    def __init__(
        self,
        iterations: int = 3000,
        learning_rate: float = 0.05,
        depth: int = 6,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        seed: int | None = None,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.seed = seed
        self.model = None

    def fit(self, X: ndarray, y: ndarray) -> CatBoostUncertaintyBaseline:
        catboost = _require("catboost", "Install the bench extras with pixi before running.")
        y = _ensure_2d_y(y)
        if y.shape[1] != 1:
            raise ValueError("CatBoostUncertaintyBaseline currently supports one-dimensional y.")
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y[:, 0],
            test_size=0.1,
            random_state=self.seed,
        )
        self.model = catboost.CatBoostRegressor(
            loss_function="RMSEWithUncertainty",
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=self.seed,
            allow_writing_files=False,
            thread_count=self.thread_count,
            verbose=False,
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )
        return self

    def sample(self, X: ndarray, n_samples: int = 200, seed: int | None = None, **kwargs) -> ndarray:
        del kwargs
        rng = np.random.default_rng(seed)
        pred = np.asarray(self.model.predict(X, prediction_type="RMSEWithUncertainty"))
        mean = pred[:, 0]
        variance = np.maximum(pred[:, -1], 1e-12)
        return rng.normal(mean, np.sqrt(variance), size=(n_samples, X.shape[0]))[:, :, None]


BASELINE_BUILDERS = {
    "ngboost": NGBoostGaussianBaseline,
    "ibug": IBUGXGBoostBaseline,
    "drf": DistributionalRandomForestBaseline,
    "qreg_lightgbm": LightGBMQuantileBaseline,
    "deep_ensemble": DeepEnsembleBaseline,
    "card": CARDRegressionBaseline,
    "catboost_uncertainty": CatBoostUncertaintyBaseline,
}


def make_baseline_model(model_type: str, params: dict[str, Any], seed: int):
    if model_type not in BASELINE_BUILDERS:
        available = ", ".join(sorted(BASELINE_BUILDERS))
        raise ValueError(f"Unknown benchmark baseline {model_type!r}. Available: {available}.")
    kwargs = dict(params)
    kwargs["seed"] = seed
    return BASELINE_BUILDERS[model_type](**kwargs)
