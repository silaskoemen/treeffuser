from __future__ import annotations

import csv
import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.datasets import make_dataset
from benchmarks.metrics import evaluate_samples
from benchmarks.variants import Variant
from benchmarks.variants import make_variants


@dataclass(frozen=True)
class ResolvedSeeds:
    data_seed: int
    model_seed: int
    sampler_seed: int


@dataclass(frozen=True)
class SamplerConfig:
    n_samples: int
    n_steps: int
    n_parallel: int


def run_benchmark(config: dict[str, Any], output_path: Path) -> None:
    variants = make_variants(config["variants"])
    datasets = config["datasets"]
    seeds = config["seeds"]
    sampler_configs = _make_sampler_configs(config["samplers"])
    seed_policy = config.get("seed_policy", {})
    provenance = get_provenance()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_config in datasets:
        for seed in seeds:
            resolved_seeds = resolve_seeds(seed=seed, seed_policy=seed_policy)
            dataset = make_dataset(
                name=dataset_config["name"],
                n_train=dataset_config["n_train"],
                n_test=dataset_config["n_test"],
                x_dim=dataset_config.get("x_dim", 3),
                seed=resolved_seeds.data_seed,
            )
            for variant in variants:
                model, fit_time = fit_variant(
                    variant=variant,
                    X_train=dataset.X_train,
                    y_train=dataset.y_train,
                    model_seed=resolved_seeds.model_seed,
                )
                for sampler_config in sampler_configs:
                    row = evaluate_variant(
                        model=model,
                        variant=variant,
                        dataset_name=dataset.name,
                        dataset_config=dataset_config,
                        X_test=dataset.X_test,
                        y_test=dataset.y_test,
                        seeds=resolved_seeds,
                        sampler_config=sampler_config,
                        fit_time=fit_time,
                        provenance=provenance,
                    )
                    rows.append(row)
                    write_rows(output_path, rows)


def fit_variant(variant, X_train, y_train, model_seed: int):
    model = variant.make_model(seed=model_seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    return model, fit_time


def evaluate_variant(
    model,
    variant: Variant,
    dataset_name: str,
    dataset_config: dict[str, Any],
    X_test,
    y_test,
    seeds: ResolvedSeeds,
    sampler_config: SamplerConfig,
    fit_time: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    y_samples = model.sample(
        X_test,
        n_samples=sampler_config.n_samples,
        n_parallel=sampler_config.n_parallel,
        n_steps=sampler_config.n_steps,
        seed=seeds.sampler_seed,
        verbose=False,
    )
    sample_time = time.perf_counter() - start

    metrics = evaluate_samples(y_samples=y_samples, y_true=y_test, X_test=X_test)
    n_repeats = variant.params.get("n_repeats", None)
    training_rows = dataset_config["n_train"] * n_repeats if n_repeats is not None else None

    row = {
        "dataset": dataset_name,
        "variant": variant.name,
        "seed": seeds.data_seed,
        "data_seed": seeds.data_seed,
        "model_seed": seeds.model_seed,
        "sampler_seed": seeds.sampler_seed,
        "n_train": dataset_config["n_train"],
        "n_test": dataset_config["n_test"],
        "x_dim": dataset_config.get("x_dim", 3),
        "y_dim": y_test.shape[1],
        "n_samples": sampler_config.n_samples,
        "n_steps": sampler_config.n_steps,
        "n_parallel": sampler_config.n_parallel,
        "fit_time": fit_time,
        "sample_time": sample_time,
        "training_rows": training_rows,
        "n_estimators_true": _format_n_estimators_true(model),
        "n_estimators_true_sum": _sum_n_estimators_true(model),
        "n_estimators_true_max": _max_n_estimators_true(model),
    }
    row.update(provenance)
    row.update(metrics)
    row.update({f"param_{key}": value for key, value in sorted(variant.params.items())})
    return row


def resolve_seeds(seed: int, seed_policy: dict[str, Any]) -> ResolvedSeeds:
    data_offset = int(seed_policy.get("data_seed_offset", 0))
    model_offset = int(seed_policy.get("model_seed_offset", 10_000))
    sampler_offset = int(seed_policy.get("sampler_seed_offset", 20_000))
    return ResolvedSeeds(
        data_seed=seed + data_offset,
        model_seed=seed + model_offset,
        sampler_seed=seed + sampler_offset,
    )


def get_provenance() -> dict[str, Any]:
    return {
        "git_sha": _run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_run_git(["status", "--short"])),
        "treeffuser_source_hash": _hash_treeffuser_source(),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_sampler_configs(config: list[dict[str, Any]]) -> list[SamplerConfig]:
    return [
        SamplerConfig(
            n_samples=int(item["n_samples"]),
            n_steps=int(item["n_steps"]),
            n_parallel=int(item.get("n_parallel", 10)),
        )
        for item in config
    ]


def _format_n_estimators_true(model) -> str:
    values = _get_n_estimators_true(model)
    if values is None:
        return ""
    if isinstance(values, list):
        return "|".join(str(value) for value in values)
    return str(values)


def _sum_n_estimators_true(model) -> int | str:
    values = _get_n_estimators_true(model)
    if values is None:
        return ""
    if isinstance(values, list):
        return sum(values)
    return values


def _max_n_estimators_true(model) -> int | str:
    values = _get_n_estimators_true(model)
    if values is None:
        return ""
    if isinstance(values, list):
        return max(values)
    return values


def _get_n_estimators_true(model):
    values = getattr(model, "n_estimators_true", None)
    if values is None and hasattr(model, "model"):
        values = getattr(model.model, "n_estimators_true", None)
    return values


def _run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def _hash_treeffuser_source() -> str:
    root = Path(__file__).resolve().parents[1] / "src" / "treeffuser"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
