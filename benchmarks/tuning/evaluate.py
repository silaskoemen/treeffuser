from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from benchmarks.datasets import make_dataset
from benchmarks.harness import get_provenance
from benchmarks.metrics import crps_climatology
from benchmarks.metrics import crps_skill_score
from benchmarks.metrics import evaluate_samples
from benchmarks.tuning.search_spaces import SPACES
from benchmarks.tuning.splits import build_splits
from benchmarks.tuning.study import _space_fingerprint
from benchmarks.variants import Variant

DEFAULT_RESULTS_DIR = Path("benchmarks/results/tuning_eval")


def evaluate_tuned_yaml(
    tuned_yaml: Path,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    strict_space_match: bool = True,
) -> Path:
    """Run K-1 eval folds on the params persisted in a tuning YAML.

    Writes one JSONL row per (eval_fold). Returns the output path.
    """
    payload = yaml.safe_load(tuned_yaml.read_text())
    protocol = payload["protocol"]
    space_name = protocol["space_name"]
    if space_name not in SPACES:
        raise ValueError(f"Tuned YAML references unknown space {space_name!r}.")
    space = SPACES[space_name]

    if strict_space_match:
        current_fp = _space_fingerprint(space)
        if current_fp != protocol["space_fingerprint"]:
            raise RuntimeError(
                f"Search space {space_name!r} has changed since this YAML was written "
                f"(stored={protocol['space_fingerprint']!r}, current={current_fp!r}). "
                f"Re-run tuning before evaluating, or pass --no-strict-space."
            )

    dataset_name = protocol["dataset"]
    bound = logger.bind(dataset=dataset_name, space=space_name)
    bound.info("Loading tuned config: {}", tuned_yaml)

    bundle = make_dataset(
        name=dataset_name,
        n_train=protocol["n_train"],
        n_test=protocol["n_test"],
        seed=protocol["master_seed"],
        x_dim=protocol["x_dim"],
    )
    X = np.concatenate([bundle.X_train, bundle.X_test], axis=0)
    y = np.concatenate([bundle.y_train, bundle.y_test], axis=0)
    splits = build_splits(
        X,
        y,
        master_seed=protocol["master_seed"],
        n_folds=protocol["n_folds"],
        name=dataset_name,
    )
    bound.info(
        "Splits rebuilt: n_total={}, K={}, eval_folds={}",
        X.shape[0],
        protocol["n_folds"],
        protocol["n_folds"] - 1,
    )

    params = payload["params"]
    sampler = payload["sampler"]
    provenance = get_provenance()

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset_name}__{space_name}.jsonl"
    out_path.unlink(missing_ok=True)

    base_sample_kwargs = _sample_kwargs(sampler)
    fold_indices = list(range(protocol["n_folds"]))
    eval_fold_ids = [k for k in fold_indices if k != 0]

    with out_path.open("a") as fh:
        for eval_fold, fold in zip(eval_fold_ids, splits.eval_folds, strict=True):
            X_train, y_train = splits.slice(fold.train_idx)
            X_test, y_test = splits.slice(fold.test_idx)

            fold_bound = bound.bind(eval_fold=eval_fold)
            fold_bound.info(
                "Fit start (n_train={}, n_test={})",
                X_train.shape[0],
                X_test.shape[0],
            )

            variant = Variant(name=space_name, params=params, model=space.model)
            model = variant.make_model(seed=protocol["master_seed"] + 10_000)

            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_samples = model.sample(
                X_test,
                seed=protocol["master_seed"] + 20_000 + eval_fold,
                **base_sample_kwargs,
            )
            sample_time = time.perf_counter() - t0

            metrics = evaluate_samples(y_samples=y_samples, y_true=y_test, X_test=X_test)
            clim = crps_climatology(y_train=y_train, y_true=y_test)

            row: dict[str, Any] = {
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "dataset": dataset_name,
                "space": space_name,
                "model": space.model,
                "eval_fold": eval_fold,
                "master_seed": protocol["master_seed"],
                "n_folds": protocol["n_folds"],
                "n_train_fold": int(X_train.shape[0]),
                "n_test_fold": int(X_test.shape[0]),
                "fit_time": fit_time,
                "sample_time": sample_time,
                "crps_climatology": clim,
                "crps_skill_score": crps_skill_score(crps_model=metrics["crps"], crps_climatology_val=clim),
                "protocol_fingerprint": payload["protocol_fingerprint"],
                "best_value_crps_tuning": payload["best_value_crps"],
                "sampler": sampler,
                "params": params,
            }
            row.update(provenance)
            row.update(metrics)
            fh.write(json.dumps(_json_safe(row), sort_keys=True))
            fh.write("\n")
            fh.flush()
            fold_bound.success(
                "CRPS={:.4f} (skill={:.3f}); fit={:.1f}s, sample={:.1f}s",
                metrics["crps"],
                row["crps_skill_score"],
                fit_time,
                sample_time,
            )

    bound.success("Wrote {} eval rows to {}", len(eval_fold_ids), out_path)
    return out_path


def _sample_kwargs(sampler: dict[str, Any] | None) -> dict[str, Any]:
    if sampler is None:
        return {"n_samples": 200}
    return {
        "n_samples": sampler.get("n_samples", 200),
        "n_parallel": sampler.get("n_parallel", 10),
        "n_steps": sampler.get("n_steps", 25),
        "verbose": False,
        "sampler_method": sampler.get("method", "euler"),
        "pf_ode": sampler.get("pf_ode", False),
        "velocity_stochasticity": sampler.get("velocity_stochasticity", 0.0),
        "velocity_stochasticity_schedule": sampler.get("velocity_stochasticity_schedule", "linear"),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run K-1 eval folds for a tuned-params YAML.")
    parser.add_argument("tuned_yaml", type=Path, nargs="+", help="One or more tuned-params YAML files")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--no-strict-space",
        dest="strict_space_match",
        action="store_false",
        help="Skip the search-space fingerprint check (re-evaluates stale tunings).",
    )
    args = parser.parse_args(argv)

    _setup_logging()

    for path in args.tuned_yaml:
        evaluate_tuned_yaml(
            path,
            results_dir=args.results_dir,
            strict_space_match=args.strict_space_match,
        )


def _setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=("<green>{time:HH:mm:ss}</green> " "<level>{level: <7}</level> " "<cyan>[{extra}]</cyan> " "{message}"),
    )


if __name__ == "__main__":
    main()
