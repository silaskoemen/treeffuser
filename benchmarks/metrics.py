from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from jaxtyping import Float

DEFAULT_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
VALID_WIDTH_COVERAGE_ERROR_TOLERANCES = (0.01, 0.02)


DIFFICULTY_BIN_KEYS = ("iqr", "std", "crps")


def evaluate_samples(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
    X_test: Float[np.ndarray, "batch x_dim"],
    coverage_levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
    n_x_bins: int = 5,
    n_difficulty_bins: int = 5,
) -> dict[str, Any]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_samples.ndim == 2:
        y_samples = y_samples[:, :, None]

    y_mean = y_samples.mean(axis=0)
    residual = y_mean - y_true
    per_point_crps_vec = per_point_crps(y_samples, y_true)
    pit_stats = pit_ks_test(y_samples, y_true)

    result = {
        "crps": float(np.mean(per_point_crps_vec)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "dss": dawid_sebastiani_score(y_samples, y_true),
        "quantile_rmsce": quantile_calibration_error(y_samples, y_true)["rmsce"],
        "quantile_mace": quantile_calibration_error(y_samples, y_true)["mace"],
        "pit_ks_stat": pit_stats["pit_ks_stat"],
        "pit_ks_pvalue": pit_stats["pit_ks_pvalue"],
    }

    bin_keys = difficulty_bin_keys(y_samples, per_point_crps_vec)

    for level in coverage_levels:
        coverage_stats = interval_stats(y_samples, y_true, level=level)
        prefix = f"interval_{int(level * 100)}"
        result[f"{prefix}_coverage"] = coverage_stats["coverage"]
        coverage_error = coverage_stats["coverage"] - level
        abs_coverage_error = abs(coverage_error)
        result[f"{prefix}_coverage_error"] = coverage_error
        result[f"{prefix}_abs_coverage_error"] = abs_coverage_error
        result[f"{prefix}_width"] = coverage_stats["width"]
        for tolerance in VALID_WIDTH_COVERAGE_ERROR_TOLERANCES:
            tolerance_pct = int(tolerance * 100)
            result[f"{prefix}_valid_width_{tolerance_pct:02d}"] = (
                coverage_stats["width"] if abs_coverage_error <= tolerance else None
            )

        by_bin = coverage_by_x_bin(
            y_samples=y_samples,
            y_true=y_true,
            X_test=X_test,
            level=level,
            n_bins=n_x_bins,
        )
        result[f"{prefix}_xbin_mace"] = by_bin["mace"]
        result[f"{prefix}_xbin_max_error"] = by_bin["max_error"]

        lower = np.quantile(y_samples, (1.0 - level) / 2.0, axis=0)
        upper = np.quantile(y_samples, 1.0 - (1.0 - level) / 2.0, axis=0)
        covered = ((y_true >= lower) & (y_true <= upper)).mean(axis=1)
        widths = (upper - lower).mean(axis=1)
        for bin_name, key in bin_keys.items():
            stats = binned_coverage_and_crps(
                bin_key=key,
                covered=covered,
                widths=widths,
                per_point_crps_vec=per_point_crps_vec,
                level=level,
                n_bins=n_difficulty_bins,
            )
            result[f"{prefix}_{bin_name}bin_coverages"] = stats["coverages"]
            result[f"{prefix}_{bin_name}bin_widths"] = stats["widths"]
            result[f"{prefix}_{bin_name}bin_crps_means"] = stats["crps_means"]
            result[f"{prefix}_{bin_name}bin_counts"] = stats["counts"]
            result[f"{prefix}_{bin_name}bin_mace"] = stats["mace"]
            result[f"{prefix}_{bin_name}bin_max_error"] = stats["max_error"]

    return result


def dawid_sebastiani_score(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
) -> float:
    """Mean Dawid-Sebastiani score over batch and y_dim.

    DSS_i = log(var_i) + (y_i - mu_i)^2 / var_i, with mu_i, var_i the per-point
    predictive mean and variance from `y_samples`. Penalises mis-calibrated
    variance more sharply than CRPS; commonly reported alongside CRPS to
    disentangle mean vs variance miscalibration. Variance is floored at 1e-12
    to avoid log(0) when the model produces a Dirac.
    """
    samples = np.asarray(y_samples)
    truth = np.asarray(y_true)
    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)
    if samples.ndim == 2:
        samples = samples[:, :, None]
    mu = samples.mean(axis=0)
    var = np.maximum(samples.var(axis=0, ddof=1), 1e-12)
    dss = np.log(var) + (truth - mu) ** 2 / var
    return float(np.mean(dss))


def pit_ks_test(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
) -> dict[str, float]:
    """Kolmogorov-Smirnov calibration test of PIT values against Uniform(0,1).

    Under perfect calibration, PIT_i = F_i(y_i) ~ Uniform(0,1) where F_i is
    the model's predictive CDF at point i. We use the sample-based PIT
    PIT_i = mean(y_samples_i <= y_i). Values are pooled across output dimensions
    before the KS test. Returns the (two-sided) KS statistic and p-value;
    larger p-values fail to reject the uniform null at standard levels.
    """
    from scipy.stats import kstest

    samples = np.asarray(y_samples)
    truth = np.asarray(y_true)
    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)
    if samples.ndim == 2:
        samples = samples[:, :, None]
    pit = np.mean(samples <= truth[None, :, :], axis=0).reshape(-1)
    result = kstest(pit, "uniform")
    return {"pit_ks_stat": float(result.statistic), "pit_ks_pvalue": float(result.pvalue)}


def crps_climatology(
    y_train: Float[np.ndarray, "n_train y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
) -> float:
    """
    Mean CRPS of the empirical marginal distribution of `y_train` evaluated at
    `y_true`. This is the "climatological forecast" reference for CRPS skill
    scores: a model that knows nothing about `x` and predicts the marginal y
    distribution from training data. CRPS_model / CRPS_climatology = 1 means
    the model adds no information beyond the marginal.
    """
    y_train_arr = np.asarray(y_train)
    y_true_arr = np.asarray(y_true)
    if y_train_arr.ndim == 1:
        y_train_arr = y_train_arr.reshape(-1, 1)
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    n_train = y_train_arr.shape[0]
    n_test = y_true_arr.shape[0]
    y_dim = y_true_arr.shape[1]
    # Broadcast training samples across the test batch and reuse the per-point
    # CRPS estimator. This handles the (n_samples, batch, y_dim) layout already.
    samples = np.broadcast_to(y_train_arr[:, None, :], (n_train, n_test, y_dim))
    return float(np.mean(per_point_crps(samples, y_true_arr)))


def crps_skill_score(crps_model: float, crps_climatology_val: float) -> float:
    """CRPSS = 1 - CRPS_model / CRPS_climatology. 1=perfect, 0=marginal, <0=worse."""
    if crps_climatology_val <= 0:
        return float("nan")
    return 1.0 - crps_model / crps_climatology_val


def per_point_crps(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    y_true: Float[np.ndarray, "batch y_dim"],
) -> Float[np.ndarray, "batch"]:
    """Per-test-point CRPS, averaged over y_dim. Same formula as `crps_ensemble`, no batch mean."""
    samples = np.asarray(y_samples)
    truth = np.asarray(y_true)
    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)
    mean_abs_error = np.mean(np.abs(samples - truth[None, :, :]), axis=0)
    sorted_samples = np.sort(samples, axis=0)
    n_samples = samples.shape[0]
    weights = 2 * np.arange(1, n_samples + 1).reshape(-1, 1, 1) - n_samples - 1
    pairwise_abs = 2.0 * np.sum(weights * sorted_samples, axis=0) / (n_samples**2)
    per_point = mean_abs_error - 0.5 * pairwise_abs
    return per_point.mean(axis=1)


def difficulty_bin_keys(
    y_samples: Float[np.ndarray, "n_samples batch y_dim"],
    per_point_crps_vec: Float[np.ndarray, "batch"],
) -> dict[str, np.ndarray]:
    """Per-point binning keys averaged over y_dim.

    Two of these — `iqr` (75th-25th sample quantile) and `std` (sample std) — are
    *predicted-uncertainty* keys derived from `y_samples` alone. The third, `crps`,
    is *outcome-conditioned*: per-point CRPS depends on `y_true` and therefore
    measures actual difficulty rather than the model's belief about it. Treat
    `iqr`/`std`-binned coverage as a calibration diagnostic and `crps`-binned
    coverage as a failure-mode diagnostic; the two answer different questions."""
    q25 = np.quantile(y_samples, 0.25, axis=0)
    q75 = np.quantile(y_samples, 0.75, axis=0)
    iqr = (q75 - q25).mean(axis=1)
    std = y_samples.std(axis=0, ddof=1).mean(axis=1)
    return {"iqr": iqr, "std": std, "crps": per_point_crps_vec}


def binned_coverage_and_crps(
    bin_key: Float[np.ndarray, "batch"],
    covered: Float[np.ndarray, "batch"],
    widths: Float[np.ndarray, "batch"],
    per_point_crps_vec: Float[np.ndarray, "batch"],
    level: float,
    n_bins: int,
) -> dict[str, Any]:
    """Stratify by `bin_key` (per-point predicted scalar). Report per-bin coverage, width, CRPS."""
    n = bin_key.shape[0]
    edges = np.quantile(bin_key, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    coverages: list[float] = []
    widths_out: list[float] = []
    crps_means: list[float] = []
    counts: list[int] = []
    if edges.shape[0] <= 2:
        coverages.append(float(np.mean(covered)))
        widths_out.append(float(np.mean(widths)))
        crps_means.append(float(np.mean(per_point_crps_vec)))
        counts.append(int(n))
    else:
        for low, high in pairwise(edges):
            if high == edges[-1]:
                mask = (bin_key >= low) & (bin_key <= high)
            else:
                mask = (bin_key >= low) & (bin_key < high)
            if np.any(mask):
                coverages.append(float(np.mean(covered[mask])))
                widths_out.append(float(np.mean(widths[mask])))
                crps_means.append(float(np.mean(per_point_crps_vec[mask])))
                counts.append(int(np.sum(mask)))
    errors = [abs(c - level) for c in coverages]
    return {
        "coverages": coverages,
        "widths": widths_out,
        "crps_means": crps_means,
        "counts": counts,
        "mace": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
    }


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
