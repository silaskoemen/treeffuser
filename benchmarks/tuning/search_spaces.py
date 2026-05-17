"""Per-model tunable surfaces, fixed method-defining params, and bound samplers.

Contract:
- `tunable(trial)` returns the dict of hyperparameters sampled by Optuna for one trial.
- `fixed` is a dict of method-defining hyperparameters that MUST NOT vary across trials
  or between dataset folds.
- `sampler` is the sampler-config dict bound to the variant (None for non-treeffuser
  baselines that produce samples without an external sampler config).
- The model is instantiated with `{**fixed, **trial_params}`. Keys in `tunable` and
  `fixed` must be disjoint; this is asserted at registration time.

For the Treeffuser family, all three headline variants (published, score+, FM-VP-ODE)
share an identical tunable LightGBM surface so the comparison is on the method, not
the search budget. Variant-specific design choices live in `fixed` and `sampler`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import optuna

TrialParams = dict[str, Any]
TunableFn = Callable[[optuna.Trial], TrialParams]


@dataclass(frozen=True)
class SearchSpace:
    model: str  # value passed to Variant.model
    tunable: TunableFn
    fixed: dict[str, Any] = field(default_factory=dict)
    sampler: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        sample = self.tunable(_DryRunTrial())
        overlap = set(sample) & set(self.fixed)
        if overlap:
            raise ValueError(f"Search space for {self.model!r} has params in both tunable and fixed: {overlap}")

    def materialize(self, trial: optuna.Trial) -> TrialParams:
        return {**self.fixed, **self.tunable(trial)}


class _DryRunTrial:
    """Minimal Optuna-trial-shaped stub for the disjointness check in __post_init__."""

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False, step: float | None = None
    ) -> float:
        return low

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False, step: int = 1) -> int:
        return low

    def suggest_categorical(self, name: str, choices):
        return next(iter(choices))


# ---------------------------------------------------------------------------
# Shared Treeffuser LightGBM tunable surface
# ---------------------------------------------------------------------------


def _treeffuser_lgbm_tunable(trial: optuna.Trial) -> TrialParams:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10]),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


_TREEFFUSER_LGBM_FIXED = {
    "early_stopping_rounds": 50,
    "n_repeats": 30,
    "subsample_freq": 1,
    "verbose": -1,
}


# Residualizer-C: the residualizer LightGBM config locked by the sweep in
# Appendix C of the paper and reused by Score+ and FM. Not tied to the main
# LightGBM tuning surface — fixed across all (model, dataset, trial) tuples.
_RESIDUALIZER_C = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 63,
    "min_child_samples": 10,
}


# ---------------------------------------------------------------------------
# Treeffuser variants
# ---------------------------------------------------------------------------

TREEFFUSER_PUBLISHED = SearchSpace(
    model="treeffuser",
    tunable=_treeffuser_lgbm_tunable,
    fixed={
        **_TREEFFUSER_LGBM_FIXED,
        "training_objective": "score",
        "score_parameterization": "noise",
        "noise_features": "raw_time",
        "t_sampling": "uniform",
        "residualize": "off",
        "sde_name": "vesde",
    },
    sampler={
        "n_samples": 200,
        "n_parallel": 20,
        "n_steps": 50,
        "method": "euler",
        "pf_ode": False,
    },
)


TREEFFUSER_SCORE_PLUS = SearchSpace(
    model="treeffuser",
    tunable=_treeffuser_lgbm_tunable,
    fixed={
        **_TREEFFUSER_LGBM_FIXED,
        "training_objective": "score",
        "score_parameterization": "edm",
        "noise_features": "raw_time_log_std",
        "t_sampling": "log_sigma_normal",
        "log_sigma_p_mean": -1.2,
        "log_sigma_p_std": 1.2,
        "residualize": "mean",
        "residualize_k_folds": 5,
        "extra_residualizer_params": _RESIDUALIZER_C,
        "sde_name": "vesde",
        "sde_hyperparam_min": 0.01,
        "sde_hyperparam_max": 20.0,
    },
    sampler={
        "n_samples": 200,
        "n_parallel": 20,
        "n_steps": 25,
        "method": "heun",
        "pf_ode": True,
    },
)


TREEFFUSER_FM = SearchSpace(
    model="treeffuser",
    tunable=_treeffuser_lgbm_tunable,
    fixed={
        **_TREEFFUSER_LGBM_FIXED,
        "training_objective": "flow_matching",
        "flow_path": "vp",
        "noise_features": "raw_time",
        "residualize": "mean",
        "residualize_k_folds": 5,
        "extra_residualizer_params": _RESIDUALIZER_C,
    },
    sampler={
        "n_samples": 200,
        "n_parallel": 20,
        "n_steps": 5,
        "method": "heun",
        "pf_ode": False,
        "velocity_stochasticity": 0.0,
        "velocity_stochasticity_schedule": "linear",
    },
)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

# Baseline ranges are centered on, and generally widen, the fixed grids in
# benchmarks/select_probabilistic_baseline_hyperparams.py. That keeps the paper
# comparison defensible: tuned baselines are not restricted to a narrower surface
# than the previously selected fixed-grid choices.

NGBOOST = SearchSpace(
    model="ngboost",
    tunable=lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 3000, log=True),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    },
    fixed={"early_stopping_rounds": 50},
)


IBUG = SearchSpace(
    model="ibug",
    tunable=lambda t: {
        "k": t.suggest_int("k", 10, 200, log=True),
        "n_estimators": t.suggest_int("n_estimators", 200, 2000, log=True),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 3, 10),
    },
    fixed={},
)


DRF = SearchSpace(
    model="drf",
    tunable=lambda t: {
        "num_trees": t.suggest_int("num_trees", 200, 2000, log=True),
        "min_node_size": t.suggest_int("min_node_size", 5, 50),
    },
    fixed={},
)


LGBM_QUANTILE = SearchSpace(
    model="qreg_lightgbm",
    tunable=lambda t: {
        "quantile_count": t.suggest_int("quantile_count", 24, 49),
        "n_estimators": t.suggest_int("n_estimators", 50, 500, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "num_leaves": t.suggest_int("num_leaves", 15, 127),
    },
    fixed={
        "min_child_samples": 20,
        "n_jobs": -1,
        "early_stopping_rounds": 30,
    },
)


DEEP_ENSEMBLE = SearchSpace(
    model="deep_ensemble",
    tunable=lambda t: {
        "hidden_size": t.suggest_categorical("hidden_size", [64, 128, 256, 512]),
        "n_layers": t.suggest_int("n_layers", 2, 5),
        "learning_rate": t.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "max_epochs": t.suggest_categorical("max_epochs", [100, 200, 400]),
    },
    fixed={"n_ensembles": 5, "patience": 25, "batch_size": 256},
)


CARD = SearchSpace(
    model="card",
    tunable=lambda t: {
        "hidden_size": t.suggest_categorical("hidden_size", [64, 128, 256]),
        "n_layers": t.suggest_int("n_layers", 2, 5),
        "learning_rate": t.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "max_epochs": t.suggest_categorical("max_epochs", [200, 400]),
        "diffusion_epochs": t.suggest_categorical("diffusion_epochs", [200, 400]),
        "n_steps": t.suggest_categorical("n_steps", [50, 100, 200]),
    },
    fixed={"beta_start": 1e-4, "beta_end": 0.02, "dropout": 0.01, "patience": 25},
)


CATBOOST = SearchSpace(
    model="catboost_uncertainty",
    tunable=lambda t: {
        "iterations": t.suggest_int("iterations", 200, 3000, log=True),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": t.suggest_int("depth", 4, 10),
    },
    fixed={"early_stopping_rounds": 50},
)


SPACES: dict[str, SearchSpace] = {
    "treeffuser_published": TREEFFUSER_PUBLISHED,
    "treeffuser_score_plus": TREEFFUSER_SCORE_PLUS,
    "treeffuser_fm": TREEFFUSER_FM,
    "ngboost": NGBOOST,
    "ibug": IBUG,
    "drf": DRF,
    "qreg_lightgbm": LGBM_QUANTILE,
    "deep_ensemble": DEEP_ENSEMBLE,
    "card": CARD,
    "catboost_uncertainty": CATBOOST,
}
