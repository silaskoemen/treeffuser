from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml
from loguru import logger

from benchmarks.datasets import make_dataset
from benchmarks.tuning.objective import TUNING_N_SAMPLES_DEFAULT
from benchmarks.tuning.objective import TuningObjective
from benchmarks.tuning.search_spaces import SPACES
from benchmarks.tuning.search_spaces import SearchSpace
from benchmarks.tuning.splits import DEFAULT_N_FOLDS
from benchmarks.tuning.splits import build_splits

DEFAULT_N_TRIALS = 25
DEFAULT_MASTER_SEED = 0
DEFAULT_RESULTS_DIR = Path("benchmarks/results/tuning")
DEFAULT_CONFIGS_DIR = Path("benchmarks/configs/tuned")
DEFAULT_MAX_ATTEMPTS_MULTIPLIER = 2
DEFAULT_PROTOCOL_VERSION = "tune-fold0-eval-folds1-5-v1"


@dataclass(frozen=True)
class Protocol:
    """Identity of a (dataset, search space, split policy) study.

    Two studies with different protocols MUST NOT share trials. The fingerprint
    is stored as an Optuna user_attr; mismatches on resume raise.
    """

    dataset: str
    space_name: str
    protocol_version: str
    n_train: int
    n_test: int
    x_dim: int
    master_seed: int
    n_folds: int
    n_samples: int
    space_fingerprint: str

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def _space_fingerprint(space: SearchSpace) -> str:
    """Hash the parts of a SearchSpace that affect trial commensurability.

    Includes `fixed`, `sampler`, `model`, and the source of `tunable` so range
    changes invalidate the cache."""
    payload = {
        "fixed": space.fixed,
        "sampler": space.sampler,
        "model": space.model,
        "tunable_source": inspect.getsource(space.tunable),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _count_trials(study: optuna.Study) -> tuple[int, int]:
    finite = 0
    failed = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.FAIL:
            failed += 1
        elif t.state == optuna.trial.TrialState.COMPLETE:
            if t.value is not None and math.isfinite(t.value):
                finite += 1
            else:
                failed += 1
    return finite, failed


def run_study(
    dataset_name: str,
    n_train: int,
    n_test: int,
    space_name: str,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
    n_trials: int = DEFAULT_N_TRIALS,
    n_samples: int = TUNING_N_SAMPLES_DEFAULT,
    max_attempts: int | None = None,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    configs_dir: Path = DEFAULT_CONFIGS_DIR,
    x_dim: int = 3,
) -> dict[str, Any]:
    """Run an Optuna study targeting `n_trials` finite trials. Each trial fits on
    fold 0's train indices and scores CRPS on fold 0's val indices.

    The study is keyed on a protocol fingerprint; resuming with mismatched
    settings raises. Failed trials (returning inf) consume attempts but not the
    finite-trial budget; the loop caps total attempts at `max_attempts`.
    """
    if space_name not in SPACES:
        raise ValueError(f"Unknown search space {space_name!r}. Available: {sorted(SPACES)}")
    space = SPACES[space_name]
    if max_attempts is None:
        max_attempts = n_trials * DEFAULT_MAX_ATTEMPTS_MULTIPLIER

    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    protocol = Protocol(
        dataset=dataset_name,
        space_name=space_name,
        protocol_version=protocol_version,
        n_train=n_train,
        n_test=n_test,
        x_dim=x_dim,
        master_seed=master_seed,
        n_folds=n_folds,
        n_samples=n_samples,
        space_fingerprint=_space_fingerprint(space),
    )
    fingerprint = protocol.fingerprint()

    bound = logger.bind(dataset=dataset_name, space=space_name)
    bound.info("Loading dataset (n_train={}, n_test={})", n_train, n_test)
    bundle = make_dataset(name=dataset_name, n_train=n_train, n_test=n_test, seed=master_seed, x_dim=x_dim)
    X = np.concatenate([bundle.X_train, bundle.X_test], axis=0)
    y = np.concatenate([bundle.y_train, bundle.y_test], axis=0)
    splits = build_splits(X, y, master_seed=master_seed, n_folds=n_folds, name=dataset_name)
    bound.info(
        "Splits: n_total={}, K={}, tuning_train={}, tuning_val={}, eval_folds={}",
        X.shape[0],
        n_folds,
        splits.tuning.train_idx.size,
        splits.tuning.val_idx.size,
        n_folds - 1,
    )

    storage_path = (results_dir / f"{dataset_name}__{space_name}__{fingerprint}.db").resolve()
    storage = f"sqlite:///{storage_path}"
    study_name = f"{dataset_name}__{space_name}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=master_seed, n_startup_trials=8)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    existing = study.user_attrs.get("protocol_fingerprint")
    if existing is None:
        study.set_user_attr("protocol_fingerprint", fingerprint)
        study.set_user_attr("protocol", asdict(protocol))
    elif existing != fingerprint:
        raise RuntimeError(
            f"Study {study_name!r} at {storage_path} has protocol fingerprint "
            f"{existing!r}, but this run wants {fingerprint!r}. Refusing to mix "
            f"incompatible trials. Delete the .db file to start over."
        )

    objective = TuningObjective(
        space_name=space_name,
        space=space,
        splits=splits,
        model_seed=master_seed + 10_000,
        sampler_seed=master_seed + 20_000,
        n_samples=n_samples,
    )

    while True:
        finite, failed = _count_trials(study)
        remaining = n_trials - finite
        total = finite + failed
        if remaining <= 0:
            bound.info("Reached target: {} finite trials ({} failed).", finite, failed)
            break
        if total >= max_attempts:
            bound.warning(
                "Hit max attempts cap ({}); have {} finite + {} failed, still {} short of target.",
                max_attempts,
                finite,
                failed,
                remaining,
            )
            break
        batch = min(remaining, max_attempts - total)
        bound.info(
            "Running batch of {} trials (have {} finite / {} failed; target {}).", batch, finite, failed, n_trials
        )
        study.optimize(objective, n_trials=batch, show_progress_bar=False)

    finite, failed = _count_trials(study)
    if finite == 0:
        raise RuntimeError(
            f"Study {study_name!r} produced 0 finite trials in {failed} failed attempts. "
            f"Not writing a results YAML. Inspect {storage_path} and the trial logs."
        )

    if not math.isfinite(study.best_value):
        raise RuntimeError(
            f"Study {study_name!r} best_value is non-finite ({study.best_value!r}) "
            f"despite {finite} finite trials reported — internal inconsistency."
        )

    best_params = {**space.fixed, **study.best_params}
    best_value = float(study.best_value)

    output = {
        "protocol": asdict(protocol),
        "protocol_fingerprint": fingerprint,
        "model": space.model,
        "best_value_crps": best_value,
        "best_tunable": dict(study.best_params),
        "params": best_params,
        "sampler": space.sampler,
        "n_trials_target": n_trials,
        "n_trials_finite": finite,
        "n_trials_failed": failed,
        "storage": str(storage_path),
    }
    out_path = configs_dir / f"{dataset_name}__{space_name}.yaml"
    out_path.write_text(yaml.safe_dump(output, sort_keys=False))
    bound.success(
        "Best CRPS={:.4f} ({} finite / {} failed trials); wrote {}",
        best_value,
        finite,
        failed,
        out_path,
    )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run per-(dataset, model) Optuna tuning study.")
    parser.add_argument("--dataset", required=True, help="Dataset name (see benchmarks/datasets.py)")
    parser.add_argument("--space", required=True, help=f"Search space name. One of: {sorted(SPACES)}")
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--x-dim", type=int, default=3)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Target finite trials")
    parser.add_argument("--max-attempts", type=int, default=None, help="Cap on total trials including failures")
    parser.add_argument("--n-samples", type=int, default=TUNING_N_SAMPLES_DEFAULT)
    parser.add_argument("--protocol-version", default=DEFAULT_PROTOCOL_VERSION)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--configs-dir", type=Path, default=DEFAULT_CONFIGS_DIR)
    args = parser.parse_args(argv)

    _setup_logging()

    run_study(
        dataset_name=args.dataset,
        n_train=args.n_train,
        n_test=args.n_test,
        space_name=args.space,
        master_seed=args.master_seed,
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        max_attempts=args.max_attempts,
        n_samples=args.n_samples,
        protocol_version=args.protocol_version,
        results_dir=args.results_dir,
        configs_dir=args.configs_dir,
        x_dim=args.x_dim,
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
