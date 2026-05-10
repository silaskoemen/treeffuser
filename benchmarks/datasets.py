from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from jaxtyping import Float


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X_train: Float[np.ndarray, "train x_dim"]
    y_train: Float[np.ndarray, "train y_dim"]
    X_test: Float[np.ndarray, "test x_dim"]
    y_test: Float[np.ndarray, "test y_dim"]


def make_dataset(
    name: str,
    n_train: int,
    n_test: int,
    seed: int,
    x_dim: int = 3,
) -> DatasetBundle:
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown benchmark dataset {name!r}. Available: {available}")

    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = rng.normal(size=(n_total, x_dim))
    y = DATASETS[name](X, rng)

    return DatasetBundle(
        name=name,
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:],
        y_test=y[n_train:],
    )


def _linear_signal(X: Float[np.ndarray, "batch x_dim"]) -> Float[np.ndarray, "batch 1"]:
    weights = np.linspace(0.7, 1.3, X.shape[1]).reshape(-1, 1)
    return X @ weights / np.sqrt(X.shape[1])


def _nonlinear_signal(X: Float[np.ndarray, "batch x_dim"]) -> Float[np.ndarray, "batch 1"]:
    linear = _linear_signal(X)
    return np.sin(linear) + 0.25 * X[:, :1] ** 2


def _heteroscedastic_scale(X: Float[np.ndarray, "batch x_dim"]) -> Float[np.ndarray, "batch 1"]:
    return 0.15 + 0.75 / (1.0 + np.exp(-1.5 * X[:, :1]))


def _homoscedastic_gaussian_linear(
    X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator
) -> Float[np.ndarray, "batch 1"]:
    mean = _linear_signal(X)
    return mean + 0.5 * rng.normal(size=mean.shape)


def _heteroscedastic_gaussian_linear(
    X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator
) -> Float[np.ndarray, "batch 1"]:
    mean = _linear_signal(X)
    scale = _heteroscedastic_scale(X)
    return mean + scale * rng.normal(size=mean.shape)


def _heteroscedastic_gaussian_nonlinear(
    X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator
) -> Float[np.ndarray, "batch 1"]:
    mean = _nonlinear_signal(X)
    scale = _heteroscedastic_scale(X)
    return mean + scale * rng.normal(size=mean.shape)


def _student_t_heavy_tail(
    X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator
) -> Float[np.ndarray, "batch 1"]:
    mean = _nonlinear_signal(X)
    scale = 0.35 + 0.25 * np.abs(X[:, :1])
    noise = rng.standard_t(df=3, size=mean.shape) / np.sqrt(3.0)
    return mean + scale * noise


def _skewed_noise(
    X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator
) -> Float[np.ndarray, "batch 1"]:
    mean = _nonlinear_signal(X)
    noise = rng.exponential(scale=1.0, size=mean.shape) - 1.0
    scale = _heteroscedastic_scale(X)
    return mean + scale * noise


def _bimodal_mixture(
    X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator
) -> Float[np.ndarray, "batch 1"]:
    base = _linear_signal(X)
    mode = np.where(rng.uniform(size=base.shape) < 0.5, -1.0, 1.0)
    separation = 0.75 + 0.25 * np.tanh(X[:, :1])
    return base + mode * separation + 0.2 * rng.normal(size=base.shape)


def _correlated_multioutput(
    X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator
) -> Float[np.ndarray, "batch 2"]:
    signal = _nonlinear_signal(X)
    mean = np.concatenate([signal, -0.5 * signal + 0.25 * X[:, :1]], axis=1)
    covariance = np.array([[0.35, 0.25], [0.25, 0.45]])
    noise = rng.multivariate_normal(mean=np.zeros(2), cov=covariance, size=X.shape[0])
    return mean + noise


DATASETS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "homoscedastic_gaussian_linear": _homoscedastic_gaussian_linear,
    "heteroscedastic_gaussian_linear": _heteroscedastic_gaussian_linear,
    "heteroscedastic_gaussian_nonlinear": _heteroscedastic_gaussian_nonlinear,
    "student_t_heavy_tail": _student_t_heavy_tail,
    "skewed_noise": _skewed_noise,
    "bimodal_mixture": _bimodal_mixture,
    "correlated_multioutput": _correlated_multioutput,
}

