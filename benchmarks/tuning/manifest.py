from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from benchmarks.tuning.evaluate import DEFAULT_RESULTS_DIR as DEFAULT_EVAL_RESULTS_DIR
from benchmarks.tuning.evaluate import evaluate_tuned_yaml
from benchmarks.tuning.objective import TUNING_N_SAMPLES_DEFAULT
from benchmarks.tuning.search_spaces import SPACES
from benchmarks.tuning.splits import DEFAULT_N_FOLDS
from benchmarks.tuning.study import DEFAULT_CONFIGS_DIR
from benchmarks.tuning.study import DEFAULT_MASTER_SEED
from benchmarks.tuning.study import DEFAULT_MAX_ATTEMPTS_MULTIPLIER
from benchmarks.tuning.study import DEFAULT_N_TRIALS
from benchmarks.tuning.study import DEFAULT_RESULTS_DIR
from benchmarks.tuning.study import run_study

DEFAULT_STATUS_PATH = Path("benchmarks/results/tuning/batch_status.jsonl")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_train: int
    n_test: int
    x_dim: int = 3


@dataclass(frozen=True)
class SpaceSpec:
    name: str
    n_trials: int | None = None
    n_samples: int | None = None
    max_attempts: int | None = None
    enabled: bool = True


@dataclass(frozen=True)
class TuningManifest:
    protocol_version: str
    master_seed: int
    n_folds: int
    n_trials: int
    n_samples: int
    max_attempts_multiplier: int
    results_dir: Path
    configs_dir: Path
    eval_results_dir: Path
    status_path: Path
    datasets: tuple[DatasetSpec, ...]
    spaces: tuple[SpaceSpec, ...]


def load_manifest(path: Path) -> TuningManifest:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest {path} must contain a top-level mapping.")

    protocol = raw.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("Manifest field 'protocol' must be a mapping.")

    datasets = tuple(_parse_dataset(item) for item in raw["datasets"])
    spaces = tuple(_parse_space(item) for item in raw["spaces"])
    if not datasets:
        raise ValueError("Manifest must list at least one dataset.")
    if not spaces:
        raise ValueError("Manifest must list at least one space.")

    return TuningManifest(
        protocol_version=str(protocol.get("version", "tune-fold0-eval-folds1-5-v1")),
        master_seed=int(protocol.get("master_seed", DEFAULT_MASTER_SEED)),
        n_folds=int(protocol.get("n_folds", DEFAULT_N_FOLDS)),
        n_trials=int(protocol.get("n_trials", DEFAULT_N_TRIALS)),
        n_samples=int(protocol.get("n_samples", TUNING_N_SAMPLES_DEFAULT)),
        max_attempts_multiplier=int(protocol.get("max_attempts_multiplier", DEFAULT_MAX_ATTEMPTS_MULTIPLIER)),
        results_dir=Path(protocol.get("results_dir", DEFAULT_RESULTS_DIR)),
        configs_dir=Path(protocol.get("configs_dir", DEFAULT_CONFIGS_DIR)),
        eval_results_dir=Path(protocol.get("eval_results_dir", DEFAULT_EVAL_RESULTS_DIR)),
        status_path=Path(protocol.get("status_path", DEFAULT_STATUS_PATH)),
        datasets=datasets,
        spaces=spaces,
    )


def run_manifest(
    manifest: TuningManifest,
    *,
    dataset_names: set[str] | None = None,
    space_names: set[str] | None = None,
    dry_run: bool = False,
    eval_after_tune: bool = False,
) -> list[dict[str, Any]]:
    jobs = list(_iter_jobs(manifest, dataset_names=dataset_names, space_names=space_names))
    if dry_run:
        for dataset, space in jobs:
            logger.info(
                "Would run dataset={} space={} n_trials={} n_samples={} max_attempts={}",
                dataset.name,
                space.name,
                _resolve(space.n_trials, manifest.n_trials),
                _resolve(space.n_samples, manifest.n_samples),
                _resolve_max_attempts(space, manifest),
            )
        return []

    manifest.status_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, space in jobs:
        bound = logger.bind(dataset=dataset.name, space=space.name)
        row: dict[str, Any] = {
            "ts_start": datetime.now(tz=timezone.utc).isoformat(),
            "dataset": dataset.name,
            "space": space.name,
            "status": "started",
        }
        t0 = time.perf_counter()
        try:
            output = run_study(
                dataset_name=dataset.name,
                n_train=dataset.n_train,
                n_test=dataset.n_test,
                space_name=space.name,
                master_seed=manifest.master_seed,
                n_folds=manifest.n_folds,
                n_trials=_resolve(space.n_trials, manifest.n_trials),
                n_samples=_resolve(space.n_samples, manifest.n_samples),
                max_attempts=_resolve_max_attempts(space, manifest),
                protocol_version=manifest.protocol_version,
                results_dir=manifest.results_dir,
                configs_dir=manifest.configs_dir,
                x_dim=dataset.x_dim,
            )
            row.update(
                {
                    "status": "ok",
                    "best_value_crps": output["best_value_crps"],
                    "n_trials_finite": output["n_trials_finite"],
                    "n_trials_failed": output["n_trials_failed"],
                    "storage": output["storage"],
                    "tuned_yaml": str(manifest.configs_dir / f"{dataset.name}__{space.name}.yaml"),
                }
            )
            if eval_after_tune:
                eval_path = evaluate_tuned_yaml(
                    Path(row["tuned_yaml"]),
                    results_dir=manifest.eval_results_dir,
                )
                row["eval_results"] = str(eval_path)
        except Exception as exc:
            row.update({"status": "failed", "error": repr(exc)})
            bound.exception("Batch job failed")
        finally:
            row["elapsed_sec"] = time.perf_counter() - t0
            row["ts_end"] = datetime.now(tz=timezone.utc).isoformat()
            _append_jsonl(manifest.status_path, row)
            rows.append(row)
    return rows


def _parse_dataset(item: dict[str, Any]) -> DatasetSpec:
    if not item.get("enabled", True):
        return DatasetSpec(name=str(item["name"]), n_train=0, n_test=0, x_dim=3)
    return DatasetSpec(
        name=str(item["name"]),
        n_train=int(item["n_train"]),
        n_test=int(item["n_test"]),
        x_dim=int(item.get("x_dim", 3)),
    )


def _parse_space(item: str | dict[str, Any]) -> SpaceSpec:
    if isinstance(item, str):
        return SpaceSpec(name=item)
    name = str(item["name"])
    if name not in SPACES:
        raise ValueError(f"Unknown search space {name!r}. Available: {sorted(SPACES)}")
    return SpaceSpec(
        name=name,
        n_trials=_optional_int(item.get("n_trials")),
        n_samples=_optional_int(item.get("n_samples")),
        max_attempts=_optional_int(item.get("max_attempts")),
        enabled=bool(item.get("enabled", True)),
    )


def _iter_jobs(
    manifest: TuningManifest,
    *,
    dataset_names: set[str] | None,
    space_names: set[str] | None,
):
    for dataset in manifest.datasets:
        if dataset.n_train <= 0 or dataset.n_test <= 0:
            continue
        if dataset_names is not None and dataset.name not in dataset_names:
            continue
        for space in manifest.spaces:
            if not space.enabled:
                continue
            if space.name not in SPACES:
                raise ValueError(f"Unknown search space {space.name!r}. Available: {sorted(SPACES)}")
            if space_names is not None and space.name not in space_names:
                continue
            yield dataset, space


def _resolve(value: int | None, default: int) -> int:
    return default if value is None else value


def _resolve_max_attempts(space: SpaceSpec, manifest: TuningManifest) -> int:
    if space.max_attempts is not None:
        return space.max_attempts
    n_trials = _resolve(space.n_trials, manifest.n_trials)
    return n_trials * manifest.max_attempts_multiplier


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(_json_safe(row), sort_keys=True))
        file.write("\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run tuning studies from a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset-name filter.")
    parser.add_argument("--spaces", nargs="+", default=None, help="Optional search-space filter.")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running studies.")
    parser.add_argument("--eval-after-tune", action="store_true", help="Run folds 1..K-1 after each tune.")
    args = parser.parse_args(argv)

    _setup_logging()
    manifest = load_manifest(args.manifest)
    rows = run_manifest(
        manifest,
        dataset_names=set(args.datasets) if args.datasets else None,
        space_names=set(args.spaces) if args.spaces else None,
        dry_run=args.dry_run,
        eval_after_tune=args.eval_after_tune,
    )
    if rows:
        failed = sum(row["status"] != "ok" for row in rows)
        if failed:
            raise SystemExit(f"{failed} tuning jobs failed; see {manifest.status_path}")


def _setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=("<green>{time:HH:mm:ss}</green> " "<level>{level: <7}</level> " "<cyan>[{extra}]</cyan> " "{message}"),
    )


if __name__ == "__main__":
    main()
