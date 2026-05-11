from __future__ import annotations

from itertools import pairwise

import numpy as np
from jaxtyping import Float

DEFAULT_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)


def evaluate_samples(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
    X_test: Float[np.ndarray, "batch x_dim"],
    coverage_levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
    n_x_bins: int = 5,
) -> dict[str, float]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_samples.ndim == 2:
        y_samples = y_samples[:, :, None]

    y_mean = y_samples.mean(axis=0)
    residual = y_mean - y_true

    result = {
        "crps": crps_ensemble(y_samples, y_true),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "quantile_rmsce": quantile_calibration_error(y_samples, y_true)["rmsce"],
        "quantile_mace": quantile_calibration_error(y_samples, y_true)["mace"],
    }

    for level in coverage_levels:
        coverage_stats = interval_stats(y_samples, y_true, level=level)
        prefix = f"interval_{int(level * 100)}"
        result[f"{prefix}_coverage"] = coverage_stats["coverage"]
        result[f"{prefix}_coverage_error"] = coverage_stats["coverage"] - level
        result[f"{prefix}_width"] = coverage_stats["width"]

        by_bin = coverage_by_x_bin(
            y_samples=y_samples,
            y_true=y_true,
            X_test=X_test,
            level=level,
            n_bins=n_x_bins,
        )
        result[f"{prefix}_xbin_mace"] = by_bin["mace"]
        result[f"{prefix}_xbin_max_error"] = by_bin["max_error"]

    return result


def crps_ensemble(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
) -> float:
    """Compute sample CRPS without depending on testbed or properscoring."""
    samples = np.asarray(y_samples)
    truth = np.asarray(y_true)
    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)

    mean_abs_error = np.mean(np.abs(samples - truth[None, :, :]), axis=0)

    sorted_samples = np.sort(samples, axis=0)
    n_samples = samples.shape[0]
    weights = 2 * np.arange(1, n_samples + 1).reshape(-1, 1, 1) - n_samples - 1
    pairwise_abs = 2.0 * np.sum(weights * sorted_samples, axis=0) / (n_samples**2)
    crps = mean_abs_error - 0.5 * pairwise_abs
    return float(np.mean(crps))


def interval_stats(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
    level: float,
) -> dict[str, float]:
    alpha = 1.0 - level
    lower = np.quantile(y_samples, alpha / 2.0, axis=0)
    upper = np.quantile(y_samples, 1.0 - alpha / 2.0, axis=0)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(np.mean(covered)),
        "width": float(np.mean(upper - lower)),
    }


def quantile_calibration_error(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
) -> dict[str, float]:
    empirical_quantiles = np.mean(y_true[None, :, :] <= y_samples, axis=0).reshape(-1)
    empirical_quantiles = np.sort(empirical_quantiles)
    expected_quantiles = np.linspace(0, 1, empirical_quantiles.shape[0])
    errors = empirical_quantiles - expected_quantiles
    return {
        "rmsce": float(np.sqrt(np.mean(errors**2))),
        "mace": float(np.mean(np.abs(errors))),
    }


def coverage_by_x_bin(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
    X_test: Float[np.ndarray, "batch x_dim"],
    level: float,
    n_bins: int,
) -> dict[str, float]:
    alpha = 1.0 - level
    lower = np.quantile(y_samples, alpha / 2.0, axis=0)
    upper = np.quantile(y_samples, 1.0 - alpha / 2.0, axis=0)
    covered = ((y_true >= lower) & (y_true <= upper)).mean(axis=1)

    feature = X_test[:, 0]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(feature, quantiles)
    edges = np.unique(edges)
    if edges.shape[0] <= 2:
        error = abs(float(np.mean(covered)) - level)
        return {"mace": error, "max_error": error}

    errors = []
    for low, high in pairwise(edges):
        if high == edges[-1]:
            mask = (feature >= low) & (feature <= high)
        else:
            mask = (feature >= low) & (feature < high)
        if np.any(mask):
            errors.append(abs(float(np.mean(covered[mask])) - level))

    return {
        "mace": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
    }
