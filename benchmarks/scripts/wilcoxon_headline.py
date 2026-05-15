"""Paired Wilcoxon signed-rank tests on the headline Treeffuser results.

Reads the per-(dataset, seed) CRPS values from `paper_real_data_v2_full.jsonl`,
averages over seeds within each (variant, dataset), and runs paired Wilcoxon
tests across the ten UCI datasets for the three pairs of interest:

    score+ vs published    (one-sided: score+ < published expected)
    FM     vs published    (one-sided: FM     < published expected)
    FM     vs score+       (two-sided: tie hypothesis)

The test unit is the dataset (n=10), not the seed. Additional seeds tighten
per-dataset means but do not change the degrees of freedom of the test.

Writes a markdown table to benchmarks/results/selected/wilcoxon_headline.md.

Usage:
    python -m benchmarks.scripts.wilcoxon_headline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "benchmarks/results/raw/paper_real_data_v2_full.jsonl"
OUTPUT_PATH = REPO_ROOT / "benchmarks/results/selected/wilcoxon_headline.md"

PUBLISHED = "treeffuser_published"
SCORE_PLUS = "treeffuser_score_combo"
FM = "vp_fm_ode_resid_C"

VARIANTS = [PUBLISHED, SCORE_PLUS, FM]
DISPLAY = {
    PUBLISHED: "published",
    SCORE_PLUS: "score+",
    FM: "FM",
}


def load_per_dataset_crps(path: Path) -> dict[str, dict[str, float]]:
    """Return {variant: {dataset: mean CRPS over seeds}}."""
    buckets: dict[str, dict[str, list[float]]] = {v: {} for v in VARIANTS}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            v = row["variant"]
            if v not in buckets:
                continue
            buckets[v].setdefault(row["dataset"], []).append(float(row["crps"]))
    return {v: {ds: float(np.mean(values)) for ds, values in by_ds.items()} for v, by_ds in buckets.items()}


def aligned_arrays(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    datasets = sorted(set(a) & set(b))
    arr_a = np.array([a[d] for d in datasets])
    arr_b = np.array([b[d] for d in datasets])
    return datasets, arr_a, arr_b


def run_pair(
    name_a: str,
    name_b: str,
    a: dict[str, float],
    b: dict[str, float],
    alternative: str,
) -> dict:
    datasets, arr_a, arr_b = aligned_arrays(a, b)
    diff = arr_a - arr_b  # negative if a < b (a better on CRPS)
    result = wilcoxon(arr_a, arr_b, alternative=alternative, zero_method="wilcox")
    return {
        "pair": f"{name_a} vs {name_b}",
        "n": len(datasets),
        "datasets": datasets,
        "diff": diff,
        "median_diff": float(np.median(diff)),
        "n_wins_a": int(np.sum(diff < 0)),
        "n_wins_b": int(np.sum(diff > 0)),
        "n_ties": int(np.sum(diff == 0)),
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "alternative": alternative,
    }


def fmt_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3f}"


def render_markdown(results: Iterable[dict], per_dataset: dict[str, dict[str, float]]) -> str:
    lines: list[str] = []
    lines.append("# Paired Wilcoxon signed-rank tests on headline CRPS\n")
    lines.append(
        "Source: `benchmarks/results/raw/paper_real_data_v2_full.jsonl`. "
        "Each variant's CRPS is averaged over 3 seeds per dataset; the paired "
        "test is run across the ten UCI datasets (n=10). Lower CRPS is better, "
        "so a negative signed difference favours the left-hand variant.\n"
    )

    lines.append("## Summary\n")
    lines.append("| Pair | Alt. | n | a-wins | b-wins | ties | median Δ | W | p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['pair']} | {r['alternative']} | {r['n']} | "
            f"{r['n_wins_a']} | {r['n_wins_b']} | {r['n_ties']} | "
            f"{r['median_diff']:+.4f} | {r['statistic']:.1f} | {fmt_p(r['pvalue'])} |"
        )
    lines.append("")

    lines.append("## Per-dataset mean CRPS (3-seed average)\n")
    datasets = sorted({d for v in per_dataset.values() for d in v})
    header_cells = ["dataset"] + [DISPLAY[v] for v in VARIANTS]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for ds in datasets:
        cells = [ds] + [f"{per_dataset[v].get(ds, float('nan')):.4f}" for v in VARIANTS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-dataset signed differences\n")
    for r in results:
        lines.append(f"### {r['pair']} (alt={r['alternative']})\n")
        lines.append("| dataset | Δ CRPS |")
        lines.append("|---|---:|")
        for ds, d in zip(r["datasets"], r["diff"]):
            lines.append(f"| {ds} | {d:+.4f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    per_dataset = load_per_dataset_crps(INPUT_PATH)
    for v in VARIANTS:
        n_ds = len(per_dataset[v])
        if n_ds != 10:
            raise RuntimeError(f"Expected 10 datasets for variant {v}, got {n_ds}: " f"{sorted(per_dataset[v])}")

    results = [
        run_pair(
            DISPLAY[SCORE_PLUS],
            DISPLAY[PUBLISHED],
            per_dataset[SCORE_PLUS],
            per_dataset[PUBLISHED],
            alternative="less",
        ),
        run_pair(
            DISPLAY[FM],
            DISPLAY[PUBLISHED],
            per_dataset[FM],
            per_dataset[PUBLISHED],
            alternative="less",
        ),
        run_pair(
            DISPLAY[FM],
            DISPLAY[SCORE_PLUS],
            per_dataset[FM],
            per_dataset[SCORE_PLUS],
            alternative="two-sided",
        ),
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_markdown(results, per_dataset))

    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    for r in results:
        print(
            f"  {r['pair']:<24} alt={r['alternative']:<9} "
            f"W={r['statistic']:>6.1f}  p={fmt_p(r['pvalue']):<8}  "
            f"a-wins={r['n_wins_a']}  b-wins={r['n_wins_b']}"
        )


if __name__ == "__main__":
    main()
