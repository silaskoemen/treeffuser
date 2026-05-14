from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from jaxtyping import Float
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_diabetes


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
    if name in REAL_DATASETS:
        return REAL_DATASETS[name](n_train=n_train, n_test=n_test, seed=seed)

    if name not in DATASETS:
        available = ", ".join(sorted([*DATASETS, *REAL_DATASETS]))
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


def _skewed_noise(X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator) -> Float[np.ndarray, "batch 1"]:
    mean = _nonlinear_signal(X)
    noise = rng.exponential(scale=1.0, size=mean.shape) - 1.0
    scale = _heteroscedastic_scale(X)
    return mean + scale * noise


def _bimodal_mixture(X: Float[np.ndarray, "batch x_dim"], rng: np.random.Generator) -> Float[np.ndarray, "batch 1"]:
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


def _diabetes(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    data = load_diabetes()
    X = data.data
    y = data.target.reshape(-1, 1)
    return _split_real_dataset(name="diabetes", X=X, y=y, n_train=n_train, n_test=n_test, seed=seed)


def _california_housing(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1)
    return _split_real_dataset(name="california_housing", X=X, y=y, n_train=n_train, n_test=n_test, seed=seed)


def _kin8nm(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    data = fetch_openml(name="kin8nm", version=1, as_frame=True, parser="auto")
    X = data.data.to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.float64).reshape(-1, 1)
    return _split_real_dataset(name="kin8nm", X=X, y=y, n_train=n_train, n_test=n_test, seed=seed)


def _wine_quality_white(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    data = fetch_openml(name="wine-quality-white", version=1, as_frame=True, parser="auto")
    X = data.data.to_numpy(dtype=np.float64)
    y = data.target.astype(int).to_numpy(dtype=np.float64).reshape(-1, 1)
    return _split_real_dataset(name="wine_quality_white", X=X, y=y, n_train=n_train, n_test=n_test, seed=seed)


def _split_real_dataset(
    name: str,
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    n_train: int,
    n_test: int,
    seed: int,
) -> DatasetBundle:
    n_total = n_train + n_test
    if n_total > X.shape[0]:
        raise ValueError(f"Dataset {name!r} has {X.shape[0]} rows, but {n_total} were requested.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])[:n_total]
    X = X[idx]
    y = y[idx]
    return DatasetBundle(
        name=name,
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:],
        y_test=y[n_train:],
    )


def _load_testbed_dataset(name: str, n_train: int, n_test: int, seed: int) -> DatasetBundle:
    """Load a preprocessed dataset from the local `testbed` package."""
    testbed_src = Path(__file__).resolve().parents[1] / "testbed" / "src"
    if str(testbed_src) not in sys.path:
        sys.path.insert(0, str(testbed_src))
    from testbed.data.utils import get_data

    d = get_data(name, verbose=False)
    X = np.asarray(d["x"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] > 1:
        y = y[:, :1]
    return _split_real_dataset(name=name, X=X, y=y, n_train=n_train, n_test=n_test, seed=seed)


def _load_ucimlrepo_dataset(
    name: str, uci_id: int, target_col: str, n_train: int, n_test: int, seed: int
) -> DatasetBundle:
    """Load a UCI dataset via the `ucimlrepo` API. Picks a single target column."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=uci_id)
    X = ds.data.features.to_numpy(dtype=np.float64)
    y = ds.data.targets[target_col].to_numpy(dtype=np.float64).reshape(-1, 1)
    return _split_real_dataset(name=name, X=X, y=y, n_train=n_train, n_test=n_test, seed=seed)


def _yacht(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    return _load_testbed_dataset("yacht", n_train, n_test, seed)


def _concrete(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    return _load_ucimlrepo_dataset(
        "concrete",
        uci_id=165,
        target_col="Concrete compressive strength",
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )


def _energy(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    # Two targets: Y1 (heating load) and Y2 (cooling load). Use Y1 by paper convention.
    return _load_ucimlrepo_dataset(
        "energy",
        uci_id=242,
        target_col="Y1",
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )


def _wine(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    # Testbed `wine` is the combined red+white set with a categorical color flag (idx 11).
    # We treat it as a continuous feature; LightGBM can still split on it cleanly.
    return _load_testbed_dataset("wine", n_train, n_test, seed)


def _power_plant(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    return _load_ucimlrepo_dataset(
        "power_plant",
        uci_id=294,
        target_col="PE",
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )


def _naval(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    return _load_testbed_dataset("naval", n_train, n_test, seed)


def _protein(n_train: int, n_test: int, seed: int) -> DatasetBundle:
    return _load_testbed_dataset("protein", n_train, n_test, seed)


REAL_DATASETS = {
    # Pre-existing
    "diabetes": _diabetes,
    "california_housing": _california_housing,
    "kin8nm": _kin8nm,
    "wine_quality_white": _wine_quality_white,
    # Canonical UCI probabilistic-regression benchmarks
    "yacht": _yacht,
    "concrete": _concrete,
    "energy": _energy,
    "wine": _wine,
    "power_plant": _power_plant,
    "naval": _naval,
    "protein": _protein,
}
