from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.datasets import make_dataset
from benchmarks.metrics import DEFAULT_COVERAGE_LEVELS
from benchmarks.metrics import binned_coverage_and_crps
from benchmarks.metrics import difficulty_bin_keys
from benchmarks.metrics import evaluate_samples
from benchmarks.metrics import per_point_crps
from benchmarks.variants import Variant
from benchmarks.variants import make_variants
from treeffuser._conformal import ConformalQuantileCalibrator


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
    method: str = "euler"
    pf_ode: bool = False
    velocity_stochasticity: float = 0.0
    variants: tuple[str, ...] | None = None


def run_benchmark(config: dict[str, Any], output_path: Path, output_format: str = "jsonl") -> None:
    variants = make_variants(config["variants"])
    datasets = config["datasets"]
    seeds = config["seeds"]
    sampler_configs = _make_sampler_configs(config["samplers"])
    seed_policy = config.get("seed_policy", {})
    conformal_cal_fraction = config.get("conformal_cal_fraction", None)
    provenance = get_provenance()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ResultWriter(output_path, output_format)
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
                    if sampler_config.variants is not None and variant.name not in sampler_config.variants:
                        continue
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
                        conformal_cal_fraction=conformal_cal_fraction,
                    )
                    writer.write(row)


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
    conformal_cal_fraction: float | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    y_samples = model.sample(
        X_test,
        n_samples=sampler_config.n_samples,
        n_parallel=sampler_config.n_parallel,
        n_steps=sampler_config.n_steps,
        seed=seeds.sampler_seed,
        verbose=False,
        sampler_method=sampler_config.method,
        pf_ode=sampler_config.pf_ode,
        velocity_stochasticity=sampler_config.velocity_stochasticity,
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
        "sampler_method": sampler_config.method,
        "sampler_pf_ode": sampler_config.pf_ode,
        "sampler_velocity_stochasticity": sampler_config.velocity_stochasticity,
        "fit_time": fit_time,
        "sample_time": sample_time,
        "training_rows": training_rows,
        "n_estimators_true": _format_n_estimators_true(model),
        "n_estimators_true_sum": _sum_n_estimators_true(model),
        "n_estimators_true_max": _max_n_estimators_true(model),
        "residualizer_mean_oof_mse": _residualizer_mean_oof_mse(model),
    }
    row.update(provenance)
    row.update(metrics)
    if conformal_cal_fraction is not None:
        row.update(
            _conformal_metrics(
                y_samples=y_samples,
                y_test=y_test,
                cal_fraction=conformal_cal_fraction,
                cal_seed=seeds.sampler_seed + 1,
            )
        )
    row.update({f"param_{key}": value for key, value in sorted(variant.params.items())})
    return row


def _conformal_metrics(
    y_samples,
    y_test,
    cal_fraction: float,
    cal_seed: int,
) -> dict[str, Any]:
    """Split-CQR coverage and width on a held-out calibration slice of the test set."""
    if not 0.0 < cal_fraction < 1.0:
        raise ValueError("conformal_cal_fraction must be in (0, 1).")
    n = y_test.shape[0]
    n_cal = round(n * cal_fraction)
    if n_cal < 2 or n - n_cal < 1:
        raise ValueError(f"conformal_cal_fraction={cal_fraction} leaves an unusable split for n_test={n}.")
    rng = np.random.default_rng(cal_seed)
    perm = rng.permutation(n)
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
    samples_cal = y_samples[:, cal_idx]
    samples_eval = y_samples[:, eval_idx]
    y_cal = y_test[cal_idx]
    y_eval = y_test[eval_idx]

    if samples_eval.ndim == 2:
        samples_eval_3d = samples_eval[:, :, None]
    else:
        samples_eval_3d = samples_eval
    y_eval_2d = y_eval.reshape(-1, 1) if y_eval.ndim == 1 else y_eval
    per_point_crps_eval = per_point_crps(samples_eval_3d, y_eval_2d)
    bin_keys = difficulty_bin_keys(samples_eval_3d, per_point_crps_eval)

    out: dict[str, Any] = {"conformal_n_cal": int(n_cal), "conformal_n_eval": int(n - n_cal)}
    for level in DEFAULT_COVERAGE_LEVELS:
        cal = ConformalQuantileCalibrator(level=level).fit_from_samples(samples_cal, y_cal)
        lower, upper = cal.predict_interval_from_samples(samples_eval)
        covered = (y_eval_2d >= lower) & (y_eval_2d <= upper)
        coverage = float(np.mean(covered))
        width = float(np.mean(upper - lower))
        prefix = f"conformal_interval_{int(level * 100)}"
        out[f"{prefix}_coverage"] = coverage
        out[f"{prefix}_coverage_error"] = coverage - level
        out[f"{prefix}_abs_coverage_error"] = abs(coverage - level)
        out[f"{prefix}_width"] = width
        radius = cal.radius
        out[f"{prefix}_cal_radius"] = float(np.mean(radius)) if radius is not None else None

        covered_pp = covered.mean(axis=1)
        widths_pp = (upper - lower).mean(axis=1)
        for bin_name, key in bin_keys.items():
            stats = binned_coverage_and_crps(
                bin_key=key,
                covered=covered_pp,
                widths=widths_pp,
                per_point_crps_vec=per_point_crps_eval,
                level=level,
                n_bins=5,
            )
            out[f"{prefix}_{bin_name}bin_coverages"] = stats["coverages"]
            out[f"{prefix}_{bin_name}bin_widths"] = stats["widths"]
            out[f"{prefix}_{bin_name}bin_crps_means"] = stats["crps_means"]
            out[f"{prefix}_{bin_name}bin_counts"] = stats["counts"]
            out[f"{prefix}_{bin_name}bin_mace"] = stats["mace"]
            out[f"{prefix}_{bin_name}bin_max_error"] = stats["max_error"]
    return out


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


class ResultWriter:
    def __init__(self, path: Path, output_format: str) -> None:
        if output_format not in {"jsonl", "csv"}:
            raise ValueError("output_format must be 'jsonl' or 'csv'.")
        self.path = path
        self.output_format = output_format
        self.rows: list[dict[str, Any]] = []
        self.path.unlink(missing_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if self.output_format == "jsonl":
            with self.path.open("a") as file:
                file.write(json.dumps(_json_safe(row), sort_keys=True))
                file.write("\n")
        else:
            self._rewrite_csv()

    def _rewrite_csv(self) -> None:
        fieldnames = sorted({key for row in self.rows for key in row})
        with self.path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _make_sampler_configs(config: list[dict[str, Any]]) -> list[SamplerConfig]:
    return [
        SamplerConfig(
            n_samples=int(item["n_samples"]),
            n_steps=int(item["n_steps"]),
            n_parallel=int(item.get("n_parallel", 10)),
            method=str(item.get("method", "euler")),
            pf_ode=bool(item.get("pf_ode", False)),
            velocity_stochasticity=float(item.get("velocity_stochasticity", 0.0)),
            variants=tuple(item["variants"]) if "variants" in item else None,
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


def _residualizer_mean_oof_mse(model) -> float | None:
    residualizer = getattr(model, "_residualizer", None)
    if residualizer is None:
        return None
    return getattr(residualizer, "mean_oof_mse", None)


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
