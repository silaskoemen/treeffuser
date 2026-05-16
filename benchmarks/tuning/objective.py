from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import optuna
from loguru import logger

from benchmarks.metrics import evaluate_samples
from benchmarks.tuning.search_spaces import SearchSpace
from benchmarks.tuning.splits import DatasetSplits
from benchmarks.variants import Variant

TUNING_N_SAMPLES_DEFAULT = 100


@dataclass(frozen=True)
class TuningObjective:
    """Optuna objective that fits one trial on the tuning train split and
    scores CRPS on the tuning val split (fold 0).

    Failed trials (any exception in fit/sample) are logged and return +inf so
    the study keeps running rather than aborting.
    """

    space_name: str
    space: SearchSpace
    splits: DatasetSplits
    model_seed: int
    sampler_seed: int
    n_samples: int = TUNING_N_SAMPLES_DEFAULT

    def __call__(self, trial: optuna.Trial) -> float:
        bound = logger.bind(
            dataset=self.splits.name,
            space=self.space_name,
            trial=trial.number,
        )

        try:
            params = self.space.materialize(trial)
        except Exception as exc:
            bound.exception("Param materialization failed: {}", exc)
            return float("inf")

        variant = Variant(name=self.space_name, params=params, model=self.space.model)
        X_train, y_train = self.splits.slice(self.splits.tuning.train_idx)
        X_val, y_val = self.splits.slice(self.splits.tuning.val_idx)

        bound.info(
            "Trial start (n_train={}, n_val={})",
            X_train.shape[0],
            X_val.shape[0],
        )
        bound.debug("Tunable params: {}", dict(trial.params))

        try:
            t_fit = time.perf_counter()
            model = variant.make_model(seed=self.model_seed)
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t_fit

            t_sample = time.perf_counter()
            y_samples = model.sample(
                X_val,
                **_sample_kwargs(self.space.sampler, self.n_samples, self.sampler_seed),
            )
            sample_time = time.perf_counter() - t_sample
        except Exception as exc:
            bound.warning("Trial failed during fit/sample: {}", exc)
            return float("inf")

        metrics = evaluate_samples(y_samples=y_samples, y_true=y_val, X_test=X_val)
        crps = float(metrics["crps"])
        bound.info(
            "Trial done: CRPS={:.4f}, fit={:.1f}s, sample={:.1f}s",
            crps,
            fit_time,
            sample_time,
        )
        return crps


def _sample_kwargs(sampler: dict[str, Any] | None, n_samples: int, seed: int) -> dict[str, Any]:
    """Sample-call kwargs. For Treeffuser variants, pull the bound sampler config
    from the search space; for baselines (sampler=None), pass minimal args.

    Tuning uses n_samples from the call site (typically < the final eval count) but
    inherits all other knobs from the bound sampler so the inference path matches.
    """
    if sampler is None:
        return {"n_samples": n_samples, "seed": seed}
    return {
        "n_samples": n_samples,
        "n_parallel": sampler.get("n_parallel", 10),
        "n_steps": sampler.get("n_steps", 25),
        "seed": seed,
        "verbose": False,
        "sampler_method": sampler.get("method", "euler"),
        "pf_ode": sampler.get("pf_ode", False),
        "velocity_stochasticity": sampler.get("velocity_stochasticity", 0.0),
        "velocity_stochasticity_schedule": sampler.get("velocity_stochasticity_schedule", "linear"),
    }
