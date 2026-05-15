import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.run import load_config

SEED_POLICY = {
    "data_seed_offset": 0,
    "model_seed_offset": 10000,
    "sampler_seed_offset": 20000,
}

SELECTION_SEEDS = [0, 1, 2]

SELECTION_DATASETS = [
    {"name": "heteroscedastic_gaussian_linear", "n_train": 1000, "n_test": 1000, "x_dim": 3},
    {"name": "heteroscedastic_gaussian_nonlinear", "n_train": 1000, "n_test": 1000, "x_dim": 3},
    {"name": "student_t_heavy_tail", "n_train": 1000, "n_test": 1000, "x_dim": 3},
    {"name": "bimodal_mixture", "n_train": 1000, "n_test": 1000, "x_dim": 3},
    {"name": "diabetes", "n_train": 300, "n_test": 142},
    {"name": "california_housing", "n_train": 500, "n_test": 300},
    {"name": "kin8nm", "n_train": 500, "n_test": 300},
    {"name": "wine_quality_white", "n_train": 500, "n_test": 300},
]

PAPER_DATASETS = [
    {"name": "yacht", "n_train": 277, "n_test": 31},
    {"name": "concrete", "n_train": 927, "n_test": 103},
    {"name": "energy", "n_train": 691, "n_test": 77},
    {"name": "diabetes", "n_train": 400, "n_test": 42},
    {"name": "wine", "n_train": 5847, "n_test": 650},
    {"name": "kin8nm", "n_train": 7372, "n_test": 820},
    {"name": "power_plant", "n_train": 8611, "n_test": 957},
    {"name": "naval", "n_train": 10741, "n_test": 1193},
    {"name": "california_housing", "n_train": 18576, "n_test": 2064},
    {"name": "protein", "n_train": 41157, "n_test": 4573},
]

FAMILY_GRIDS = {
    "ngboost": {
        "model": "ngboost",
        "constant": {"early_stopping_rounds": 50},
        "grid": {
            "n_estimators": [1000, 3000],
            "learning_rate": [0.02, 0.05, 0.1],
        },
    },
    "ibug": {
        "model": "ibug",
        "constant": {"learning_rate": 0.05, "leaf_sample_trees": 64},
        "grid": {
            "k": [50, 100, 200],
            "n_estimators": [300, 1000],
            "max_depth": [3, 6],
        },
    },
    "qreg_lightgbm": {
        "model": "qreg_lightgbm",
        "constant": {"n_estimators": 500, "min_child_samples": 20, "n_jobs": -1, "early_stopping_rounds": 30},
        "grid": {
            "quantile_count": [21, 31],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31],
        },
    },
    "deep_ensemble": {
        "model": "deep_ensemble",
        "constant": {"max_epochs": 150, "learning_rate": 0.001, "batch_size": 256, "patience": 15},
        "grid": {
            "n_ensembles": [3, 5],
            "hidden_size": [50, 100],
            "n_layers": [1, 2],
        },
    },
    "card": {
        "model": "card",
        "constant": {
            "max_epochs": 150,
            "diffusion_epochs": 150,
            "learning_rate": 0.001,
            "batch_size": 256,
            "patience": 15,
            "dropout": 0.01,
            "sample_batch_size": 4096,
        },
        "grid": {
            "hidden_size": [50, 100],
            "n_layers": [1],
            "n_steps": [50, 100],
        },
    },
    "catboost_uncertainty": {
        "model": "catboost_uncertainty",
        "constant": {"iterations": 3000, "early_stopping_rounds": 50, "thread_count": -1},
        "grid": {
            "learning_rate": [0.03, 0.05, 0.1],
            "depth": [4, 6, 8],
        },
    },
}


def main() -> None:
    args = parse_args()
    if args.command == "write-configs":
        write_selection_configs(args)
    elif args.command == "select":
        select_hyperparams(args)
    else:
        raise ValueError(f"Unknown command {args.command!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and select external probabilistic baseline hyperparameter sweeps. "
            "This is intentionally separate from the full paper performance config."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write-configs", help="Write one benchmark config per model family.")
    write_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/configs/probabilistic_baseline_hyperparams"),
        help="Directory for generated family-specific YAML configs.",
    )
    write_parser.add_argument(
        "--families",
        nargs="+",
        default=sorted(FAMILY_GRIDS),
        choices=sorted(FAMILY_GRIDS),
        help="Model families to generate.",
    )

    select_parser = subparsers.add_parser("select", help="Select one winner per family from completed JSONL/CSV runs.")
    select_parser.add_argument(
        "--results", nargs="+", type=Path, required=True, help="Selection result JSONL/CSV files."
    )
    select_parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        required=True,
        help="Generated selection configs used for the runs. Needed to recover exact params.",
    )
    select_parser.add_argument(
        "--metric",
        default="crps",
        help="Primary selection metric. Lower is better unless --maximize is set.",
    )
    select_parser.add_argument(
        "--tie-breaker",
        default="interval_90_abs_coverage_error",
        help="Secondary metric for ties or near-ties. Lower is better.",
    )
    select_parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.0025,
        help="Relative primary-metric tolerance within which the tie-breaker decides.",
    )
    select_parser.add_argument(
        "--min-completeness",
        type=float,
        default=1.0,
        help="Required fraction of the most complete variant's rows within each family.",
    )
    select_parser.add_argument("--maximize", action="store_true", help="Maximize the primary metric.")
    select_parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("benchmarks/results/selected/probabilistic_baseline_hyperparams.json"),
        help="JSON file containing rankings and selected variants.",
    )
    select_parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("benchmarks/results/selected/probabilistic_baseline_hyperparams.md"),
        help="Human-readable selection summary.",
    )
    select_parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("benchmarks/configs/paper_probabilistic_baselines_selected.yaml"),
        help="Final external-baseline benchmark config containing selected winners only.",
    )
    return parser.parse_args()


def write_selection_configs(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_commands = []
    for family in args.families:
        variants = make_grid_variants(family)
        config = {
            "seeds": SELECTION_SEEDS,
            "seed_policy": SEED_POLICY,
            "datasets": SELECTION_DATASETS,
            "variants": variants,
            "samplers": [
                {
                    "n_samples": 200,
                    "n_steps": 1,
                    "n_parallel": 200,
                    "method": "euler",
                    "pf_ode": False,
                    "variants": [variant["name"] for variant in variants],
                }
            ],
        }
        path = args.output_dir / f"{family}.yaml"
        write_yaml(path, config, header=selection_config_header(family))
        print(f"Wrote {path} ({len(variants)} variants)")
        run_commands.append(
            "pixi run python -m benchmarks.run "
            f"--config {path} "
            f"--output benchmarks/results/raw/probabilistic_baseline_hyperparams_{family}.jsonl"
        )
    print("\nRun the selection sweeps before calling `select`:")
    for command in run_commands:
        print(f"  {command}")


def make_grid_variants(family: str) -> list[dict[str, Any]]:
    spec = FAMILY_GRIDS[family]
    variants = []
    keys = list(spec["grid"])
    for values in itertools.product(*(spec["grid"][key] for key in keys)):
        params = dict(spec["constant"])
        params.update(dict(zip(keys, values, strict=True)))
        variants.append(
            {
                "name": f"{family}__{slugify_params(params)}",
                "model": spec["model"],
                "params": params,
            }
        )
    return variants


def select_hyperparams(args: argparse.Namespace) -> None:
    result_paths = expand_paths(args.results, label="result")
    config_paths = expand_paths(args.configs, label="config")
    rows = read_rows(result_paths)
    variant_specs = read_variant_specs(config_paths)
    rankings, winners = rank_families(
        rows=rows,
        variant_specs=variant_specs,
        metric=args.metric,
        tie_breaker=args.tie_breaker,
        tie_tolerance=args.tie_tolerance,
        min_completeness=args.min_completeness,
        maximize=args.maximize,
    )
    summary = {
        "protocol": {
            "selection_suite": {
                "seeds": SELECTION_SEEDS,
                "datasets": SELECTION_DATASETS,
                "aggregation": "mean over seeds within dataset, then unweighted mean over datasets",
            },
            "metric": args.metric,
            "metric_direction": "max" if args.maximize else "min",
            "tie_breaker": args.tie_breaker,
            "tie_tolerance": args.tie_tolerance,
            "min_completeness": args.min_completeness,
            "results": [str(path) for path in result_paths],
            "configs": [str(path) for path in config_paths],
        },
        "winners": winners,
        "rankings": rankings,
    }
    write_json(args.output_summary, summary)
    write_markdown(args.output_markdown, summary)
    write_final_config(args.output_config, winners)
    print(f"Wrote selection summary to {args.output_summary}")
    print(f"Wrote selection report to {args.output_markdown}")
    print(f"Wrote selected paper config to {args.output_config}")


def expand_paths(paths: list[Path], label: str) -> list[Path]:
    expanded = []
    for path in paths:
        raw = str(path)
        if any(char in raw for char in "*?["):
            expanded.extend(sorted(Path().glob(raw)))
        else:
            expanded.append(path)
    missing = [path for path in expanded if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing {label} file(s): {joined}")
    if not expanded:
        raise FileNotFoundError(
            f"No {label} files matched. Run the generated hyperparameter-selection benchmark configs first."
        )
    return expanded


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if path.suffix == ".csv":
            with path.open(newline="") as file:
                rows.extend(dict(row) for row in csv.DictReader(file))
        else:
            with path.open() as file:
                for line in file:
                    stripped = line.strip()
                    if stripped:
                        rows.append(json.loads(stripped))
    if not rows:
        raise ValueError("No result rows found.")
    return rows


def read_variant_specs(paths: list[Path]) -> dict[str, dict[str, Any]]:
    specs = {}
    for path in paths:
        config = load_config(path)
        for variant in config["variants"]:
            if not variant.get("enabled", True):
                continue
            name = variant["name"]
            if name in specs:
                raise ValueError(f"Variant {name!r} appears in more than one config.")
            specs[name] = {
                "name": name,
                "model": variant["model"],
                "params": variant["params"],
                "config": str(path),
            }
    return specs


def rank_families(
    rows: list[dict[str, Any]],
    variant_specs: dict[str, dict[str, Any]],
    metric: str,
    tie_breaker: str,
    tie_tolerance: float,
    min_completeness: float,
    maximize: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        variant = row["variant"]
        if variant not in variant_specs:
            raise ValueError(f"Result row references variant {variant!r}, but it is missing from --configs.")
        family = variant_specs[variant]["model"]
        grouped[(family, variant)].append(row)

    specs_by_family = defaultdict(list)
    for spec in variant_specs.values():
        specs_by_family[spec["model"]].append(spec)
    rankings = {}
    winners = {}
    for family, specs in sorted(specs_by_family.items()):
        row_counts = {spec["name"]: len(grouped[(family, spec["name"])]) for spec in specs}
        max_rows = max(row_counts.values())
        if max_rows == 0:
            raise ValueError(f"No result rows found for any {family} candidate.")
        min_rows = math.ceil(max_rows * min_completeness)
        incomplete = {variant: count for variant, count in row_counts.items() if count < min_rows}
        if incomplete and min_completeness >= 1.0:
            raise ValueError(
                f"Incomplete {family} selection run. Missing or partial candidates: {incomplete}. "
                "Finish the run, or lower --min-completeness to explicitly exclude partial candidates."
            )

        summaries = []
        for spec in specs:
            variant_rows = grouped[(family, spec["name"])]
            if len(variant_rows) < min_rows:
                continue
            summaries.append(
                summarize_variant(
                    rows=variant_rows,
                    metric=metric,
                    tie_breaker=tie_breaker,
                    spec=spec,
                )
            )
        complete = [summary for summary in summaries if not math.isnan(summary["score"])]
        if not complete:
            raise ValueError(f"No complete candidates for {family}; lower --min-completeness or inspect failed runs.")
        ranked = sorted(
            complete,
            key=lambda summary: rank_key(summary=summary, maximize=maximize),
        )
        winner = choose_winner(ranked=ranked, tie_tolerance=tie_tolerance, maximize=maximize)
        rankings[family] = ranked
        winners[family] = {
            "variant": winner["variant"],
            "model": winner["model"],
            "params": winner["params"],
            "score": winner["score"],
            "tie_breaker_score": winner["tie_breaker_score"],
            "row_count": winner["row_count"],
            "dataset_count": winner["dataset_count"],
            "selected_from_config": winner["config"],
        }
    return rankings, winners


def summarize_variant(
    rows: list[dict[str, Any]],
    metric: str,
    tie_breaker: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    by_dataset = defaultdict(list)
    tie_by_dataset = defaultdict(list)
    fit_times = []
    for row in rows:
        dataset = row["dataset"]
        by_dataset[dataset].append(as_float(row[metric], field=metric, variant=row["variant"]))
        tie_by_dataset[dataset].append(as_float(row[tie_breaker], field=tie_breaker, variant=row["variant"]))
        fit_times.append(as_float(row["fit_time"], field="fit_time", variant=row["variant"]))

    dataset_scores = {dataset: statistics.mean(values) for dataset, values in by_dataset.items()}
    dataset_tie_scores = {dataset: statistics.mean(values) for dataset, values in tie_by_dataset.items()}
    return {
        "variant": spec["name"],
        "model": spec["model"],
        "params": spec["params"],
        "config": spec["config"],
        "row_count": len(rows),
        "dataset_count": len(dataset_scores),
        "score": statistics.mean(dataset_scores.values()),
        "tie_breaker_score": statistics.mean(dataset_tie_scores.values()),
        "mean_fit_time": statistics.mean(fit_times),
        "dataset_scores": dataset_scores,
        "dataset_tie_breaker_scores": dataset_tie_scores,
    }


def rank_key(summary: dict[str, Any], maximize: bool) -> tuple[float, float, float]:
    primary = -summary["score"] if maximize else summary["score"]
    return primary, summary["tie_breaker_score"], summary["mean_fit_time"]


def choose_winner(
    ranked: list[dict[str, Any]],
    tie_tolerance: float,
    maximize: bool,
) -> dict[str, Any]:
    best = ranked[0]
    best_score = best["score"]
    near_ties = []
    for summary in ranked:
        if is_within_relative_tolerance(
            score=summary["score"],
            best_score=best_score,
            tolerance=tie_tolerance,
            maximize=maximize,
        ):
            near_ties.append(summary)
    return min(near_ties, key=lambda summary: (summary["tie_breaker_score"], summary["mean_fit_time"]))


def is_within_relative_tolerance(score: float, best_score: float, tolerance: float, maximize: bool) -> bool:
    denominator = max(abs(best_score), 1e-12)
    if maximize:
        return (best_score - score) / denominator <= tolerance
    return (score - best_score) / denominator <= tolerance


def write_final_config(path: Path, winners: dict[str, dict[str, Any]]) -> None:
    variants = []
    for family, winner in sorted(winners.items()):
        variants.append(
            {
                "name": f"{family}_selected",
                "model": winner["model"],
                "params": winner["params"],
            }
        )
    config = {
        "seeds": SELECTION_SEEDS,
        "seed_policy": SEED_POLICY,
        "datasets": PAPER_DATASETS,
        "variants": variants,
        "samplers": [
            {
                "n_samples": 200,
                "n_steps": 1,
                "n_parallel": 200,
                "method": "euler",
                "pf_ode": False,
                "variants": [variant["name"] for variant in variants],
            }
        ],
    }
    header = [
        "Selected external probabilistic baselines for the final paper comparison.",
        "Generated by benchmarks/select_probabilistic_baseline_hyperparams.py.",
        "Do not edit selected parameters by hand; rerun the selection script from",
        "completed hyperparameter-selection JSONL files if the protocol changes.",
        "This config intentionally excludes Treeffuser and should be joined with",
        "the already completed paper_real_data_v2 results.",
    ]
    write_yaml(path, config, header=header)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Probabilistic Baseline Hyperparameter Selection",
        "",
        "Selection suite: 4 synthetic + 4 small real datasets, 3 seeds.",
        "Aggregation: mean over seeds within each dataset, then unweighted mean over datasets.",
        f"Primary metric: `{summary['protocol']['metric']}` ({summary['protocol']['metric_direction']}).",
        f"Tie-breaker within {summary['protocol']['tie_tolerance']:.2%}: `{summary['protocol']['tie_breaker']}`.",
        "",
        "## Winners",
        "",
        "| Family | Variant | Score | Tie-breaker | Rows |",
        "|---|---|---:|---:|---:|",
    ]
    for family, winner in sorted(summary["winners"].items()):
        lines.append(
            "| "
            f"{family} | `{winner['variant']}` | {winner['score']:.6g} | "
            f"{winner['tie_breaker_score']:.6g} | {winner['row_count']} |"
        )
    lines.extend(["", "## Full Ranking", ""])
    for family, ranking in sorted(summary["rankings"].items()):
        lines.extend(
            [
                f"### {family}",
                "",
                "| Rank | Variant | Score | Tie-breaker | Fit time | Rows |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for idx, row in enumerate(ranking, start=1):
            lines.append(
                "| "
                f"{idx} | `{row['variant']}` | {row['score']:.6g} | "
                f"{row['tie_breaker_score']:.6g} | {row['mean_fit_time']:.3f} | {row['row_count']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def as_float(value: Any, field: str, variant: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse field {field!r} for variant {variant!r}: {value!r}") from exc


def selection_config_header(family: str) -> list[str]:
    return [
        f"Hyperparameter-selection sweep for {family}.",
        "This is not the final paper performance run.",
        "Run this config independently, then pass the resulting JSONL file to",
        "benchmarks/select_probabilistic_baseline_hyperparams.py select.",
        "The selection suite mirrors Treeffuser's development protocol: 4 synthetic",
        "diagnostics + 4 small real datasets, with fixed grids and paired seeds.",
    ]


def write_yaml(path: Path, data: dict[str, Any], header: list[str]) -> None:
    try:
        import yaml  # noqa: PLC0415
    except ModuleNotFoundError:
        text = simple_yaml_dump(data)
    else:

        class IndentedSafeDumper(yaml.SafeDumper):
            def increase_indent(self, flow=False, indentless=False):
                return super().increase_indent(flow=flow, indentless=False)

        text = yaml.dump(data, Dumper=IndentedSafeDumper, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"# {line}" for line in header) + "\n\n" + text)


def simple_yaml_dump(value: Any, indent: int = 0) -> str:
    space = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(simple_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{space}{key}: {format_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{space}-")
                lines.append(simple_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{space}- {format_scalar(item)}")
        return "\n".join(lines)
    return f"{space}{format_scalar(value)}"


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return str(value)


def slugify_params(params: dict[str, Any]) -> str:
    parts = []
    for key, value in sorted(params.items()):
        if key in {
            "batch_size",
            "diffusion_epochs",
            "dropout",
            "early_stopping_rounds",
            "min_child_samples",
            "n_jobs",
            "patience",
            "sample_batch_size",
            "thread_count",
        }:
            continue
        parts.append(f"{short_key(key)}{slugify_value(value)}")
    return "__".join(parts)


def short_key(key: str) -> str:
    return {
        "depth": "d",
        "hidden_size": "h",
        "iterations": "it",
        "k": "k",
        "learning_rate": "lr",
        "max_depth": "md",
        "max_epochs": "ep",
        "n_ensembles": "ens",
        "n_estimators": "est",
        "n_layers": "layers",
        "n_steps": "steps",
        "num_leaves": "leaves",
        "quantile_count": "q",
    }.get(key, key)


def slugify_value(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
