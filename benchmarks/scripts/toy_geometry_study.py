"""Toy preconditioning study for tree-based diffusion regressors.

Generates a 1D heteroscedastic conditional mixture and compares several Gaussian
probability-path parameterizations under identical LightGBM regressors. The study
isolates the first-order preconditioning effect (published-style VE noise
prediction, direct VE score prediction, raw VE-FM, preconditioned EDM, and
VP-FM): how the target scale and tree-input scale behave across t, how hard each
per-t slice is for a fresh tree to fit, and how much boosting capacity each
variant needs to reach a fixed accuracy.

The 1D setting is sufficient to resolve the preconditioning gap (orders of
magnitude in target/feature scale) but not the much smaller VP-vs-linear gap on
real tabular data; that ranking is established empirically in App G.

Outputs land in benchmarks/results/toy_geometry_study/ by default.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class StudyConfig:
    output_dir: Path
    seed: int
    n_train: int
    n_eval: int
    n_train_t: int
    n_eval_t: int
    n_estimators: int
    num_leaves: int
    sigma_min: float
    sigma_max: float
    vp_beta_min: float
    vp_beta_max: float


@dataclass(frozen=True)
class ToyData:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class VariantData:
    feature_y: np.ndarray
    noise_feature: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    builder: Callable[[ToyData, np.ndarray, StudyConfig], VariantData]


def parse_args() -> StudyConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/toy_geometry_study"),
        help="Directory for plots and summary files.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=2_500)
    parser.add_argument("--n-eval", type=int, default=4_000)
    parser.add_argument("--n-train-t", type=int, default=24)
    parser.add_argument("--n-eval-t", type=int, default=25)
    parser.add_argument("--n-estimators", type=int, default=90)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=20.0)
    parser.add_argument("--vp-beta-min", type=float, default=0.1)
    parser.add_argument("--vp-beta-max", type=float, default=20.0)
    args = parser.parse_args()
    return StudyConfig(**vars(args))


CAPACITY_T_VALUES: tuple[float, ...] = (0.1, 0.5, 0.9)
CAPACITY_N_ESTIMATORS: tuple[int, ...] = (5, 15, 45, 135, 400)


def generate_toy_data(n_samples: int, rng: np.random.Generator) -> ToyData:
    x = rng.uniform(-2.0, 2.0, size=n_samples)
    mode_probability = 1.0 / (1.0 + np.exp(-2.0 * x))
    choose_high_mode = rng.uniform(size=n_samples) < mode_probability

    low_mean = -1.4 + 0.35 * np.sin(2.2 * x)
    high_mean = 1.3 + 0.45 * np.cos(1.7 * x)
    low_scale = 0.18 + 0.14 / (1.0 + np.exp(-3.0 * (x + 0.5)))
    high_scale = 0.22 + 0.18 / (1.0 + np.exp(2.5 * (x - 0.4)))

    means = np.where(choose_high_mode, high_mean, low_mean)
    scales = np.where(choose_high_mode, high_scale, low_scale)
    y = means + scales * rng.normal(size=n_samples)
    y = (y - np.mean(y)) / np.std(y)
    z = rng.normal(size=n_samples)
    return ToyData(x=x, y=y, z=z)


def ve_sigma(t: np.ndarray, config: StudyConfig) -> tuple[np.ndarray, np.ndarray]:
    log_ratio = np.log(config.sigma_max / config.sigma_min)
    sigma = config.sigma_min * np.exp(log_ratio * t)
    sigma_dot = log_ratio * sigma
    return sigma, sigma_dot


def vp_path(t: np.ndarray, config: StudyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_prime = 0.5 * (config.vp_beta_min + t * (config.vp_beta_max - config.vp_beta_min))
    t_integral = 0.5 * config.vp_beta_min * t + 0.25 * (config.vp_beta_max - config.vp_beta_min) * t**2
    alpha_bar = np.exp(-t_integral)
    alpha = np.sqrt(alpha_bar)
    beta = np.sqrt(1.0 - alpha_bar)
    alpha_dot = -0.5 * t_prime * alpha
    beta_dot = t_prime * alpha_bar / (2.0 * np.maximum(beta, 1e-8))
    return alpha, beta, alpha_dot, beta_dot


def build_ve_score(data: ToyData, t: np.ndarray, config: StudyConfig) -> VariantData:
    sigma, _ = ve_sigma(t, config)
    y_t = data.y + sigma * data.z
    target = -data.z / sigma
    return VariantData(feature_y=y_t, noise_feature=np.log(sigma), target=target)


def build_ve_noise(data: ToyData, t: np.ndarray, config: StudyConfig) -> VariantData:
    sigma, _ = ve_sigma(t, config)
    y_t = data.y + sigma * data.z
    target = -data.z
    return VariantData(feature_y=y_t, noise_feature=np.log(sigma), target=target)


def build_ve_raw_velocity(data: ToyData, t: np.ndarray, config: StudyConfig) -> VariantData:
    sigma, sigma_dot = ve_sigma(t, config)
    y_t = data.y + sigma * data.z
    target = sigma_dot * data.z
    return VariantData(feature_y=y_t, noise_feature=np.log(sigma), target=target)


def build_ve_edm_noise(data: ToyData, t: np.ndarray, config: StudyConfig) -> VariantData:
    sigma, _ = ve_sigma(t, config)
    y_t = data.y + sigma * data.z
    c_in = 1.0 / np.sqrt(1.0 + sigma**2)
    target = data.z
    return VariantData(feature_y=c_in * y_t, noise_feature=np.log(sigma), target=target)


def build_vp_velocity(data: ToyData, t: np.ndarray, config: StudyConfig) -> VariantData:
    alpha, beta, alpha_dot, beta_dot = vp_path(t, config)
    y_t = alpha * data.y + beta * data.z
    target = alpha_dot * data.y + beta_dot * data.z
    noise_feature = np.log(np.maximum(beta, 1e-8))
    return VariantData(feature_y=y_t, noise_feature=noise_feature, target=target)


def build_linear_velocity(data: ToyData, t: np.ndarray, config: StudyConfig) -> VariantData:
    del config
    y_t = (1.0 - t) * data.y + t * data.z
    target = data.z - data.y
    noise_feature = np.log(t / (1.0 - t))
    return VariantData(feature_y=y_t, noise_feature=noise_feature, target=target)


def variants() -> list[Variant]:
    return [
        Variant("ve_direct_score", "VE direct score", build_ve_score),
        Variant("ve_noise", "VE noise/published target", build_ve_noise),
        Variant("ve_raw_velocity", "VE raw velocity", build_ve_raw_velocity),
        Variant("ve_edm_noise", "VE EDM/noise", build_ve_edm_noise),
        Variant("vp_velocity", "VP velocity", build_vp_velocity),
        Variant("linear_velocity", "Linear velocity", build_linear_velocity),
    ]


LINE_STYLES = {
    "VE direct score": {"color": "#4C78A8", "linestyle": "-", "marker": "o", "zorder": 6},
    "VE noise/published target": {"color": "#F58518", "linestyle": (0, (5, 2)), "marker": "s", "zorder": 7},
    "VE raw velocity": {"color": "#54A24B", "linestyle": (0, (1, 2)), "marker": "^", "zorder": 8},
    "VE EDM/noise": {"color": "#B279A2", "linestyle": "-.", "marker": "D", "zorder": 9},
    "VP velocity": {"color": "#E45756", "linestyle": "-", "marker": "P", "zorder": 5},
    "Linear velocity": {"color": "#72B7B2", "linestyle": "--", "marker": "X", "zorder": 4},
}


def style_for(label: str) -> dict[str, Any]:
    return LINE_STYLES[label]


def features_from_variant(data: ToyData, t: np.ndarray, variant_data: VariantData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": data.x,
            "path_y": variant_data.feature_y,
            "t": t,
            "noise_feature": variant_data.noise_feature,
        }
    )


def repeated_data(base: ToyData, t_values: np.ndarray) -> tuple[ToyData, np.ndarray]:
    n_t = len(t_values)
    repeated = ToyData(
        x=np.tile(base.x, n_t),
        y=np.tile(base.y, n_t),
        z=np.tile(base.z, n_t),
    )
    t = np.repeat(t_values, len(base.x))
    return repeated, t


def fit_model(features: pd.DataFrame, target: np.ndarray, config: StudyConfig, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=config.n_estimators,
        learning_rate=0.07,
        num_leaves=config.num_leaves,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(features, target)
    return model


def evaluate_variant(
    variant: Variant,
    train_data: ToyData,
    eval_data: ToyData,
    train_t: np.ndarray,
    eval_t: np.ndarray,
    config: StudyConfig,
) -> tuple[list[dict[str, float | str]], lgb.LGBMRegressor]:
    repeated_train_data, repeated_train_t = repeated_data(train_data, train_t)
    train_variant = variant.builder(repeated_train_data, repeated_train_t, config)
    train_features = features_from_variant(repeated_train_data, repeated_train_t, train_variant)
    model = fit_model(train_features, train_variant.target, config, config.seed)

    rows: list[dict[str, float | str]] = []
    for t_value in eval_t:
        t_array = np.full_like(eval_data.y, t_value, dtype=float)
        eval_variant = variant.builder(eval_data, t_array, config)
        eval_features = features_from_variant(eval_data, t_array, eval_variant)
        prediction = model.predict(eval_features)
        rmse = float(np.sqrt(np.mean((prediction - eval_variant.target) ** 2)))
        target_rms = float(np.sqrt(np.mean(eval_variant.target**2)))
        feature_rms = float(np.sqrt(np.mean(eval_variant.feature_y**2)))
        rows.append(
            {
                "variant": variant.key,
                "label": variant.label,
                "t": float(t_value),
                "target_rms": target_rms,
                "feature_rms": feature_rms,
                "rmse": rmse,
                "relative_rmse": rmse / target_rms,
            }
        )
    return rows, model


def evaluate_per_slice(
    variant: Variant,
    train_data: ToyData,
    eval_data: ToyData,
    eval_t: np.ndarray,
    config: StudyConfig,
) -> list[dict[str, float | str]]:
    """Fit a fresh LightGBM at each t with no t/noise feature.

    Removes cross-t capacity sharing so the difficulty of each variant's slice is
    measured intrinsically.
    """
    rows: list[dict[str, float | str]] = []
    for t_value in eval_t:
        t_train = np.full_like(train_data.y, t_value, dtype=float)
        train_variant = variant.builder(train_data, t_train, config)
        train_features = pd.DataFrame({"x": train_data.x, "path_y": train_variant.feature_y})

        t_eval = np.full_like(eval_data.y, t_value, dtype=float)
        eval_variant = variant.builder(eval_data, t_eval, config)
        eval_features = pd.DataFrame({"x": eval_data.x, "path_y": eval_variant.feature_y})

        model = fit_model(train_features, train_variant.target, config, config.seed)
        prediction = model.predict(eval_features)
        rmse = float(np.sqrt(np.mean((prediction - eval_variant.target) ** 2)))
        target_rms = float(np.sqrt(np.mean(eval_variant.target**2)))
        rows.append(
            {
                "variant": variant.key,
                "label": variant.label,
                "t": float(t_value),
                "per_slice_rmse": rmse,
                "per_slice_relative_rmse": rmse / target_rms,
            }
        )
    return rows


def capacity_sweep(
    variant: Variant,
    train_data: ToyData,
    eval_data: ToyData,
    t_values: tuple[float, ...],
    n_estimators_grid: tuple[int, ...],
    config: StudyConfig,
) -> list[dict[str, float | str]]:
    """Sweep tree-ensemble capacity at a few fixed t values.

    The supervised regression at each t is single-slice (no t/noise feature) so the
    boosting-capacity requirement is attributed cleanly to the variant's geometry.
    """
    rows: list[dict[str, float | str]] = []
    for t_value in t_values:
        t_train = np.full_like(train_data.y, t_value, dtype=float)
        train_variant = variant.builder(train_data, t_train, config)
        train_features = pd.DataFrame({"x": train_data.x, "path_y": train_variant.feature_y})

        t_eval = np.full_like(eval_data.y, t_value, dtype=float)
        eval_variant = variant.builder(eval_data, t_eval, config)
        eval_features = pd.DataFrame({"x": eval_data.x, "path_y": eval_variant.feature_y})
        target_rms = float(np.sqrt(np.mean(eval_variant.target**2)))

        for n_est in n_estimators_grid:
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=n_est,
                learning_rate=0.07,
                num_leaves=config.num_leaves,
                min_child_samples=40,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=config.seed,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(train_features, train_variant.target)
            prediction = model.predict(eval_features)
            rmse = float(np.sqrt(np.mean((prediction - eval_variant.target) ** 2)))
            rows.append(
                {
                    "variant": variant.key,
                    "label": variant.label,
                    "t": float(t_value),
                    "n_estimators": int(n_est),
                    "capacity_rmse": rmse,
                    "capacity_relative_rmse": rmse / target_rms,
                }
            )
    return rows


def plot_per_slice(per_slice: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for label, group in per_slice.groupby("label", sort=False):
        style = style_for(label)
        ax.plot(
            group["t"],
            group["per_slice_relative_rmse"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=3.2,
            label=label,
            alpha=0.8,
            zorder=style["zorder"],
        )
    ax.set_xlabel("t")
    ax.set_ylabel("Per-slice relative RMSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.04,
        "VE variants overlap here by fixed-t scalar equivalence.",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "per_slice_error_by_t.pdf")
    plt.close(fig)


def plot_capacity_sweep(capacity: pd.DataFrame, output_dir: Path) -> None:
    aggregated = capacity.groupby(["label", "n_estimators"], sort=False, as_index=False)[
        "capacity_relative_rmse"
    ].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for label, group in aggregated.groupby("label", sort=False):
        style = style_for(label)
        ax.plot(
            group["n_estimators"],
            group["capacity_relative_rmse"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=3.6,
            label=label,
            alpha=0.8,
            zorder=style["zorder"],
        )
    ax.set_xlabel("n_estimators (LightGBM rounds)")
    ax.set_ylabel("Relative RMSE (avg over t)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.text(
        0.02,
        0.04,
        "VE variants overlap here by fixed-t scalar equivalence.",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "capacity_sweep.pdf")
    plt.close(fig)


def plot_paper_figure(
    results: pd.DataFrame,
    per_slice: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Three-panel appendix figure for scale and fixed-t difficulty."""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)

    ax_target, ax_feature, ax_slice = axes
    for label, group in results.groupby("label", sort=False):
        style = style_for(label)
        ax_target.plot(
            group["t"],
            group["target_rms"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            markersize=3.0,
            label=label,
            alpha=0.8,
            zorder=style["zorder"],
        )
        ax_feature.plot(
            group["t"],
            group["feature_rms"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            markersize=3.0,
            label=label,
            alpha=0.8,
            zorder=style["zorder"],
        )
    ax_target.set_xlabel("t")
    ax_target.set_ylabel("Target RMS")
    ax_target.set_yscale("log")
    ax_target.grid(True, which="both", alpha=0.25)
    ax_target.set_title("(a) Target scale", fontsize=10)

    ax_feature.set_xlabel("t")
    ax_feature.set_ylabel("Tree-input RMS")
    ax_feature.set_yscale("log")
    ax_feature.grid(True, which="both", alpha=0.25)
    ax_feature.set_title("(b) Path feature scale", fontsize=10)

    for label, group in per_slice.groupby("label", sort=False):
        style = style_for(label)
        ax_slice.plot(
            group["t"],
            group["per_slice_relative_rmse"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            markersize=3.0,
            label=label,
            alpha=0.8,
            zorder=style["zorder"],
        )
    ax_slice.set_xlabel("t")
    ax_slice.set_ylabel("Per-slice relative RMSE")
    ax_slice.set_yscale("log")
    ax_slice.grid(True, which="both", alpha=0.25)
    ax_slice.set_title("(c) Fixed-t difficulty", fontsize=10)
    ax_slice.text(
        0.02,
        0.04,
        "VE variants overlap by fixed-t equivalence.",
        transform=ax_slice.transAxes,
        fontsize=7,
    )
    ax_slice.legend(frameon=False, fontsize=7, loc="best")

    fig.savefig(output_dir / "paper_figure.pdf")
    plt.close(fig)


def plot_scale_curves(results: pd.DataFrame, output_dir: Path) -> None:
    for metric, ylabel, filename in [
        ("target_rms", "Target RMS", "target_scale_by_t.pdf"),
        ("feature_rms", "Path feature RMS", "feature_scale_by_t.pdf"),
        ("relative_rmse", "LightGBM relative RMSE", "lightgbm_error_by_t.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        for label, group in results.groupby("label", sort=False):
            style = style_for(label)
            ax.plot(
                group["t"],
                group[metric],
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=3.2,
                label=label,
                alpha=0.8,
                zorder=style["zorder"],
            )
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if metric != "relative_rmse":
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(output_dir / filename)
        plt.close(fig)


def plot_target_slices(eval_data: ToyData, config: StudyConfig, output_dir: Path) -> None:
    shown_variants = [
        Variant("ve_direct_score", "VE direct score", build_ve_score),
        Variant("ve_noise", "VE noise/published target", build_ve_noise),
        Variant("ve_edm_noise", "VE EDM/noise", build_ve_edm_noise),
        Variant("vp_velocity", "VP velocity", build_vp_velocity),
    ]
    shown_t = np.array([0.05, 0.5, 0.95])
    rng = np.random.default_rng(config.seed + 10_000)
    sample_index = rng.choice(len(eval_data.y), size=min(1_200, len(eval_data.y)), replace=False)
    sampled = ToyData(x=eval_data.x[sample_index], y=eval_data.y[sample_index], z=eval_data.z[sample_index])

    fig, axes = plt.subplots(
        len(shown_variants),
        len(shown_t),
        figsize=(9.2, 8.0),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    for row_index, variant in enumerate(shown_variants):
        for col_index, t_value in enumerate(shown_t):
            axis = axes[row_index, col_index]
            t_array = np.full_like(sampled.y, t_value, dtype=float)
            variant_data = variant.builder(sampled, t_array, config)
            axis.scatter(variant_data.feature_y, variant_data.target, s=4, alpha=0.25, linewidths=0)
            axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
            axis.set_title(f"{variant.label}, t={t_value:.2f}", fontsize=9)
            if row_index == len(shown_variants) - 1:
                axis.set_xlabel("Regressor y-feature")
            if col_index == 0:
                axis.set_ylabel("Target")
            axis.grid(True, alpha=0.18)
    fig.savefig(output_dir / "target_slices.pdf")
    plt.close(fig)


def write_summary(
    results: pd.DataFrame,
    per_slice: pd.DataFrame,
    capacity: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path,
) -> None:
    capacity_avg = capacity.groupby(["label", "n_estimators"], sort=False, as_index=False)[
        "capacity_relative_rmse"
    ].mean()

    summary_rows = []
    for label, group in results.groupby("label", sort=False):
        target_scale_ratio = float(group["target_rms"].max() / group["target_rms"].min())
        feature_scale_ratio = float(group["feature_rms"].max() / group["feature_rms"].min())
        per_slice_group = per_slice[per_slice["label"] == label]
        capacity_curve = capacity_avg[capacity_avg["label"] == label].sort_values("n_estimators")
        summary_rows.append(
            {
                "label": label,
                "target_scale_ratio": target_scale_ratio,
                "feature_scale_ratio": feature_scale_ratio,
                "median_relative_rmse": float(group["relative_rmse"].median()),
                "max_relative_rmse": float(group["relative_rmse"].max()),
                "median_per_slice_relative_rmse": float(per_slice_group["per_slice_relative_rmse"].median()),
                "max_per_slice_relative_rmse": float(per_slice_group["per_slice_relative_rmse"].max()),
                "capacity_curve": [
                    {
                        "n_estimators": int(row["n_estimators"]),
                        "relative_rmse": float(row["capacity_relative_rmse"]),
                    }
                    for _, row in capacity_curve.iterrows()
                ],
            }
        )

    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "variants": summary_rows,
        "interpretation": (
            "Reframed as a preconditioning study. The published-style VE noise target has unit RMS, while direct "
            "VE score prediction and raw VE-FM span orders of magnitude across t. The published-style VE problem "
            "still has the unpreconditioned VE tree-input scale y0 + sigma z; EDM rescales that input through "
            "c_in, and VP-FM keeps the path feature near standardized scale by construction. The per-slice column "
            "isolates intrinsic slice difficulty by fitting a fresh LightGBM at each t with no t or noise feature; "
            "the VE curves intentionally overlap there because the targets/features differ only by fixed-t scalar "
            "maps. The capacity curve reports relative RMSE averaged over t={0.1, 0.5, 0.9} as a function of "
            "n_estimators. The 1D toy resolves preconditioning cleanly but is too low-dimensional to surface the "
            "marginal VP-vs-linear gap that real-data path-ablation runs measure (App G)."
        ),
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    train_data = generate_toy_data(config.n_train, rng)
    eval_data = generate_toy_data(config.n_eval, rng)
    train_t = np.linspace(0.02, 0.98, config.n_train_t)
    eval_t = np.linspace(0.02, 0.98, config.n_eval_t)

    result_rows: list[dict[str, float | str]] = []
    per_slice_rows: list[dict[str, float | str]] = []
    capacity_rows: list[dict[str, float | str]] = []
    for variant in variants():
        rows, _ = evaluate_variant(variant, train_data, eval_data, train_t, eval_t, config)
        result_rows.extend(rows)
        per_slice_rows.extend(evaluate_per_slice(variant, train_data, eval_data, eval_t, config))
        capacity_rows.extend(
            capacity_sweep(variant, train_data, eval_data, CAPACITY_T_VALUES, CAPACITY_N_ESTIMATORS, config)
        )

    results = pd.DataFrame(result_rows)
    per_slice = pd.DataFrame(per_slice_rows)
    capacity = pd.DataFrame(capacity_rows)

    results_path = config.output_dir / "target_scale_summary.json"
    results.to_json(results_path, orient="records", indent=2)
    per_slice.to_json(config.output_dir / "per_slice_summary.json", orient="records", indent=2)
    capacity.to_json(config.output_dir / "capacity_summary.json", orient="records", indent=2)
    write_summary(results, per_slice, capacity, config, config.output_dir)
    plot_scale_curves(results, config.output_dir)
    plot_target_slices(eval_data, config, config.output_dir)
    plot_per_slice(per_slice, config.output_dir)
    plot_capacity_sweep(capacity, config.output_dir)
    plot_paper_figure(results, per_slice, config.output_dir)

    print(f"Wrote {results_path}")
    print(f"Wrote {config.output_dir / 'summary.json'}")
    print(f"Wrote plots to {config.output_dir}")


if __name__ == "__main__":
    main()
