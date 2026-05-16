from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from benchmarks.tuning.manifest import DatasetSpec
from benchmarks.tuning.manifest import SpaceSpec
from benchmarks.tuning.manifest import TuningManifest
from benchmarks.tuning.manifest import run_manifest
from benchmarks.tuning.objective import TUNING_N_SAMPLES_DEFAULT
from benchmarks.tuning.search_spaces import SPACES
from benchmarks.tuning.splits import DEFAULT_N_FOLDS
from benchmarks.tuning.study import DEFAULT_MASTER_SEED
from benchmarks.tuning.study import DEFAULT_MAX_ATTEMPTS_MULTIPLIER

DEFAULT_PREFLIGHT_DATASET = DatasetSpec(
    name="heteroscedastic_gaussian_linear",
    n_train=400,
    n_test=100,
    x_dim=3,
)
DEFAULT_PREFLIGHT_DIR = Path("benchmarks/results/tuning_preflight")


def make_preflight_manifest(
    *,
    spaces: list[str],
    n_trials: int,
    max_attempts: int,
    n_samples: int,
    dataset: DatasetSpec = DEFAULT_PREFLIGHT_DATASET,
    output_dir: Path = DEFAULT_PREFLIGHT_DIR,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> TuningManifest:
    return TuningManifest(
        protocol_version="preflight-v1",
        master_seed=master_seed,
        n_folds=DEFAULT_N_FOLDS,
        n_trials=n_trials,
        n_samples=n_samples,
        max_attempts_multiplier=DEFAULT_MAX_ATTEMPTS_MULTIPLIER,
        results_dir=output_dir / "studies",
        configs_dir=output_dir / "configs",
        eval_results_dir=output_dir / "eval",
        status_path=output_dir / "status.jsonl",
        datasets=(dataset,),
        spaces=tuple(SpaceSpec(name=space, max_attempts=max_attempts) for space in spaces),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run small tuning preflights across search spaces.")
    parser.add_argument("--spaces", nargs="+", default=sorted(SPACES), help="Spaces to preflight.")
    parser.add_argument("--n-trials", type=int, default=2, help="Target finite trials per space.")
    parser.add_argument("--max-attempts", type=int, default=4, help="Total attempts per space, including failures.")
    parser.add_argument("--n-samples", type=int, default=min(32, TUNING_N_SAMPLES_DEFAULT))
    parser.add_argument("--n-train", type=int, default=DEFAULT_PREFLIGHT_DATASET.n_train)
    parser.add_argument("--n-test", type=int, default=DEFAULT_PREFLIGHT_DATASET.n_test)
    parser.add_argument("--x-dim", type=int, default=DEFAULT_PREFLIGHT_DATASET.x_dim)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREFLIGHT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    unknown = sorted(set(args.spaces) - set(SPACES))
    if unknown:
        raise ValueError(f"Unknown spaces {unknown}. Available: {sorted(SPACES)}")

    _setup_logging()
    manifest = make_preflight_manifest(
        spaces=args.spaces,
        n_trials=args.n_trials,
        max_attempts=args.max_attempts,
        n_samples=args.n_samples,
        dataset=DatasetSpec(
            name=DEFAULT_PREFLIGHT_DATASET.name,
            n_train=args.n_train,
            n_test=args.n_test,
            x_dim=args.x_dim,
        ),
        output_dir=args.output_dir,
        master_seed=args.master_seed,
    )
    rows = run_manifest(manifest, dry_run=args.dry_run)
    failed = sum(row["status"] != "ok" for row in rows)
    if failed:
        raise SystemExit(f"{failed} preflight jobs failed; see {manifest.status_path}")


def _setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=("<green>{time:HH:mm:ss}</green> " "<level>{level: <7}</level> " "<cyan>[{extra}]</cyan> " "{message}"),
    )


if __name__ == "__main__":
    main()
