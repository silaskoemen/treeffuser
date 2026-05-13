# Treeffuser Benchmarks

This directory contains lightweight, implementation-focused benchmarks for comparing
Treeffuser variants. It is separate from `testbed/`: `testbed/` is for broad model
comparisons, while this harness is for paired diagnostics during Treeffuser development.

The benchmark grain is one result row per:

```text
dataset x seed x variant x sampler
```

This keeps comparisons paired and makes it possible to ask whether a variant improves
coverage or CRPS without hiding the cost in wider intervals, slower sampling, or more
training rows.

## Layout

```text
benchmarks/
  run.py
  harness.py
  variants.py
  datasets.py
  metrics.py
  configs/
    smoke.yaml
    synthetic_core.yaml
    real_smoke.yaml
  results/raw/
```

## Running

The runner uses PyYAML if available and otherwise falls back to a small parser for the
simple YAML subset used by these configs:

```bash
python -m benchmarks.run --config benchmarks/configs/smoke.yaml
```

By default, results are written as JSON Lines to
`benchmarks/results/raw/<config>__<variants>_<timestamp>.jsonl`. JSONL is the preferred
format because each completed benchmark row is appended immediately and variant-specific
parameter dictionaries do not force CSV schema rewrites.

To run only selected variants from a broad config:

```bash
python -m benchmarks.run \
  --config benchmarks/configs/synthetic_core.yaml \
  --variants baseline_raw_time residualized_mean_edm_raw_time_log_std
```

CSV remains available with `--output-format csv` or by passing an `--output` path ending
in `.csv`.
The `real_smoke.yaml` config uses local datasets bundled with scikit-learn, so it does
not download external benchmark data.

## Seeding Policy

For each dataset/seed pair, the harness derives and records three seeds:

- `data_seed`: controls data generation and paired train/test splits.
- `model_seed`: controls model training randomness.
- `sampler_seed`: controls Monte Carlo sampling randomness.

The same resolved `model_seed` and `sampler_seed` are used across variants for the same
dataset/seed pair unless the config explicitly changes the offsets. This gives paired
comparisons a stable stochastic contract.

## Recommended Variant

From the 2026-05-12 decision runs (see `results/raw/synthetic_core__*_20260512_*.jsonl`
and `results/raw/real_smoke__*_20260512_*.jsonl`), the leading experimental combo is
`score_parameterization="edm"` + `residualize="mean"` + `noise_features="raw_time_log_std"`,
encoded as the `residualized_mean_edm_raw_time_log_std` variant. It ties baseline CRPS on
synthetic and beats it ~3% on real_smoke, with ~40% lower interval-90 absolute coverage
error on synthetic and ~80% lower on real_smoke. Mean residualization and EDM are
independent improvements: mean residualization mostly moves CRPS, EDM mostly moves
calibration. `mean_scale` and the plain-`x0` parameterization stayed experimental — no
clear CRPS win, higher fit cost.

The library default remains `residualize="off"` for backward compatibility.

### 2026-05-12 expanded `real_smoke` decision run

`real_smoke.yaml` now spans four datasets (`diabetes`, `california_housing`, `kin8nm`,
`wine_quality_white`). See `results/raw/real_smoke__all_20260512_194200.jsonl`. The
calibration story for `residualized_mean_edm_raw_time_log_std` holds: interval-90 absolute
coverage error drops vs `baseline_raw_time` on every dataset (avg ~50% reduction). CRPS is
mixed — wins on diabetes (-3.3%) and kin8nm (-8.9%), ties on california_housing, and loses
~5.7% on wine_quality_white. EDM-only variants tend to undercover on real data; pairing
EDM with mean residualization is what closes that gap. The next lever investigated for
CRPS uniformity is min-SNR loss weighting (see `loss_weighting`).

### 2026-05-12 min-SNR loss weighting — experimental, not adopted

Parameterization-aware min-SNR-γ weighting from Hang et al. (2023) was added behind
`loss_weighting="min_snr"` + `min_snr_gamma` and benchmarked against
`residualized_mean_edm_raw_time_log_std` with γ ∈ {1, 5}. See
`results/raw/real_smoke__*minsnr*_20260512_201658.jsonl` and
`results/raw/synthetic_core__*minsnr*_20260512_201659.jsonl`. Real data: γ=1 helps on
california_housing (CRPS −1.3%, I90 err −26%) but hurts elsewhere; γ=5 is uniformly
worse. Synthetic: γ=5 wins on bimodal_mixture (I90 err 0.014 vs 0.038) and skewed_noise
but otherwise ties or loses. Not adopted as default; surface kept opt-in for future use
on multimodal-tailed targets.

### 2026-05-12 EDM-style log-σ t sampling sweep — positive, recommended

Configurable training-time `t` distribution added behind `t_sampling="log_sigma_normal"`
with `log_sigma_p_mean` and `log_sigma_p_std`. Sweep on `residualized_mean_edm_raw_time_log_std`
across 8 datasets × 3 seeds × 5 variants. See
`results/raw/log_sigma_sweep__all_20260512_214355.jsonl`.

Result: log-σ sampling improves CRPS on **every dataset** vs `t_sampling="uniform"`. The
EDM-default `(p_mean=-1.2, p_std=1.2)` is the safest pick across all 8 sets (CRPS gains
0.5%–3.9%). Standout wins: `wine_quality_white` (the only previously regressing real
dataset) recovers a CRPS improvement under `(p_mean=-1.2, p_std=2.0)`; `bimodal_mixture`
gets the largest coverage win in the whole improvement track (interval-90 absolute
coverage error drops from 0.045 to 0.012, -74%). Tradeoff: coverage error slightly
regresses on a few small real sets (diabetes, kin8nm). This matches the GBT hypothesis
that emerged from the PF-ODE failure: tree-based score models are bin-density-limited,
so shifting `t` density beats reweighting fixed bins.

Recommended successor variant for the leading combo:
`score_parameterization="edm"` + `residualize="mean"` + `noise_features="raw_time_log_std"`
+ `t_sampling="log_sigma_normal"` (`p_mean=-1.2`, `p_std=1.2`). The library default
remains `t_sampling="uniform"` for backward compatibility; new fits should opt in.

### 2026-05-12 Residualizer-capacity sweep — high-capacity adopted for real data

Capacity sweep on the LightGBM conditional-mean model used by `residualize="mean"`,
on top of the new winning combo (EDM + log-σ t-sampling). Three points:
A — current defaults; B — regularized (shallower, more rounds, stronger min_child);
C — high-capacity (`max_depth=-1`, `num_leaves=63`, `min_child_samples=10`,
`n_estimators=300`, `learning_rate=0.05`). See
`results/raw/residualizer_sweep__all_20260512_221655.jsonl`.

Outcome: C wins on real data, A wins on synthetic. On the 4 real sets, C improves I90
absolute coverage error from `{0.044, 0.051, 0.023, 0.017}` to `{0.010, 0.010, 0.034,
0.006}` (closes the diabetes coverage regression from log-σ sampling; halves the wine
gap), and improves CRPS on 3 of 4 (slight regression only on diabetes). On synthetic,
CRPS regresses 5–11% with C — those generators have near-linear conditional means that
A already captures.

Two secondary findings worth recording:
1. **OOF MSE does not track CRPS.** C has higher OOF MSE than A on every dataset yet
   wins downstream CRPS on the real ones. The residualizer's interaction with the
   diffusion matters more than pure mean-prediction accuracy. Future residualizer
   tuning should optimize downstream CRPS via a benchmark run, not OOF MSE.
2. **Variant B (regularized) is strictly dominated.** Lower capacity doesn't help on
   any dataset; the current defaults were already on the safe side of the
   capacity axis.

Recommendation: pair the new winning combo with `extra_residualizer_params` set to the
C configuration when targeting real tabular data; keep A defaults for synthetic
diagnostics. The library defaults remain unchanged (residualizer-A) for backward
compatibility.

### 2026-05-12 Residualizer early-stopping sweep — ES helps as auto-tuner, not as a peak-quality lever

Inner-split early stopping added behind setting `early_stopping_rounds` in
`extra_residualizer_params`. The residualizer splits each fold's `train_idx` further
into 85% inner-train / 15% inner-val, with a hard gate: when inner-val < 50 rows the
residualizer warns and falls back to the empirically validated high-capacity defaults
(variant C). Sweep on A (current), C (high-cap fixed), D (ES + moderate caps), E
(ES + lifted caps). See `results/raw/residualizer_es_sweep__all_20260512_223507.jsonl`.

Findings:
1. **D is the most robust single config.** Ties or beats A on synthetic CRPS (the
   regime A used to win), and lands within 0.2–1% of C on real CRPS. Neither A nor
   C alone could span both regimes.
2. **E is dominated by D.** Lifting depth/leaf caps while running ES gains nothing —
   early stopping already controls capacity better than caps do.
3. **The size gate works cleanly.** On diabetes (inner-val = 36 < 50) the gate trips
   and D/E both fall back to C with identical results, as designed.
4. **C still wins on real-data CRPS by small margins** (0.2–0.8%). For large enough
   `n`, fixed high-capacity beats ES at moderate caps. ES's value is robustness, not
   peak performance.

Recommendation update: **C** remains the peak-quality config for real tabular data.
**D** is the new recommended "auto" config when the user does not know `n` in advance —
it self-tunes via ES on large data and falls back to C on small data via the gate.
Library defaults remain unchanged for backward compatibility.

### 2026-05-13 Conditional-coverage diagnostic — residualization is a difficulty homogenizer

Follow-up to the conformal head-to-head. Hypothesis to test: split-CQR uses a global
additive radius and could be marginally calibrated while conditionally miscalibrated;
if S1's pre-conformal intervals are properly heteroscedastic, S1+conformal should
hold coverage flatly across difficulty bins where B+conformal fails. New per-bin
diagnostics added in `benchmarks/metrics.py:binned_coverage_and_crps` and wired
through `evaluate_samples` (raw) and `harness._conformal_metrics` (conformal). Each
test row now records, for `bin_by ∈ {iqr, std, crps}` and each coverage level,
quintile-binned coverage, width, and per-bin CRPS. Re-ran the same conformal
comparison: see `results/raw/conformal_comparison__all_20260513_110233.jsonl`.

Result: the hypothesis is **rejected, and the actual finding is more interesting.**

1. **Both variants are equally well-calibrated conditionally.** IQR-bin and STD-bin
   MACE at 90% are within 0.02 of each other on every dataset. Neither B+conformal
   nor S1+conformal exhibits the "fails on hard bins" failure mode.
2. **B's predicted uncertainty tracks actual difficulty *better* than S1's.** Ratio
   (CRPS in IQR-bin5) / (CRPS in IQR-bin1), where bins are defined by each
   variant's own predicted IQR:

   | Dataset | B | S1 |
   |---|---:|---:|
   | diabetes | 1.20 | 0.97 |
   | california_housing | 3.08 | 2.02 |
   | kin8nm | 1.34 | 0.99 |
   | wine_quality_white | 1.25 | 1.00 |
   | student_t_heavy_tail | 1.63 | 1.29 |
   | bimodal_mixture | 1.41 | 1.12 |

   S1's ratios are systematically closer to 1 — the mean residualizer absorbs
   heteroscedasticity into the conditional mean, leaving the diffusion to model
   roughly homoscedastic residuals and produce roughly uniform predicted IQRs.
3. **Width-by-difficulty mirrors this.** Post-conformal width ratio bin5/bin1:
   california 2.53 (B) vs 1.69 (S1); kin8nm 1.32 vs 1.12; student_t 1.75 vs 1.08.
   B's intervals fan out by predicted difficulty; S1's are roughly uniform.
4. **The two strategies cost CRPS in different regimes.** On `kin8nm` (S1's clean
   win), per-bin CRPS shows S1 ties B on easy points and beats B 24% on hard
   points — the residualizer captured the heteroscedasticity correctly. On
   `california_housing` and `wine_quality_white` (where B wins marginal CRPS), S1
   overcautiously inflates easy-point widths without recovering the cost on hard
   points — the residualizer captured the conditional mean but not the
   heteroscedasticity it needed to homogenize residual scale.

Reframing: mean residualization is a *difficulty homogenizer*. It helps when it
captures conditional scale variation (residuals shrink to uniform magnitude
everywhere) and hurts when it captures only the conditional mean. Pure marginal
metrics can't separate these regimes — only per-bin diagnostics can.

Implication for the improvement track: the remaining headroom is not in adding
more model machinery (items #3, #8–#10 from `plans/improvements.md` are all
off-axis for what's actually happening). The open research question is
"when does residualization homogenize correctly, and can we detect it in advance?"
That's a separate workstream.

### 2026-05-13 Conformal head-to-head — model-side win is real but narrow

Direct test of whether the winning combo is genuinely a better density model or
mostly a better-calibrated one. Two variants — `B_baseline_raw_time` and
`S1_winning_combo` (EDM + mean residualization + log-σ t-sampling + raw_time_log_std
+ residualizer-C) — run on 8 datasets × 3 seeds with `conformal_cal_fraction=0.5`.
The harness already supported the wrapper via `_conformal_metrics`; this is the
first sweep to actually exercise it. See
`results/raw/conformal_comparison__all_20260513_100933.jsonl`.

Headline (width ratio S1/B at matched conformal coverage; <1 means S1 has a tighter
learned density at the same coverage level):

| Level | Raw S1/B | Conformal S1/B |
|------:|---------:|---------------:|
| Real, 50%  | 1.236 | **0.925** |
| Real, 80%  | 1.247 | 1.056 |
| Real, 90%  | 1.223 | 1.096 |
| Real, 95%  | 1.092 | 0.992 |
| Synth (50–95%) | 1.02–1.11 | 1.02–1.11 |

Findings:
1. **Most of S1's raw coverage gain was bought with wider intervals.** Real-data raw
   width is 22% larger; after conformalizing both, the gap shrinks to 5–10% at
   80–90%. Conformal alone closes ~75% of the baseline's undercoverage
   (`baseline+conformal` real |covE|@90 = 0.028 vs `S1+conformal` 0.025).
2. **S1 learns a genuinely better center.** Real 50% conformal width: S1 is 7.5%
   narrower. Mean residualization is doing real work on the conditional mode.
3. **S1's tails are too broad.** Real 80–90% conformal width slightly favors B —
   the EDM-residualized score spreads more probability mass into the wings than
   needed.
4. **kin8nm is the unambiguous win**: 24% narrower conformal width @90, 17% better
   CRPS, the only dataset where the combo decisively dominates baseline-plus-
   conformal.
5. **diabetes is noisy** (n_test=142 → cal=71/eval=71); single dataset where S1
   is meaningfully wider post-conformal — small-test instability, not a real
   regression.
6. **Synthetic regression is not fixable by conformal.** S1's CRPS is 6% worse and
   conformal width is uniformly wider — this is a property of the combo, not of
   calibration. Tracked as a known cost.

Interpretation: S1 remains the recommended config for real tabular regression on
the strength of CRPS (8.32 vs 8.45) and 50%/95% conformal width, but the
improvement-track narrative needs adjusting — for users whose only goal is interval
coverage, `baseline + conformal` is a competitive simpler pipeline. The next-step
candidates from `plans/improvements.md` (#9–#11) are not justified by this evidence;
the remaining headroom is in heteroscedastic regimes like kin8nm, where the combo
already wins clearly.

### 2026-05-13 SDE σ_max schedule sweep — null result, σ_max=20 retained

Adaptive σ_max sweep on the new winning combo (EDM + mean residualization + log-σ
t-sampling + raw_time_log_std + high-capacity residualizer-C). Five variants:
B baseline_raw_time, S1 σ_max=20 (combo default), S2 `sde_initialize_from_data=True`,
S3 σ_max=5, S4 σ_max=3. Eight datasets × 3 seeds. See
`results/raw/sde_schedule_sweep__all_20260512_231259.jsonl`.

Hypothesis: standardized residuals have std≈1, so VESDE σ_max=20 over-covers the data
scale and wastes transport at the top of the reverse-time trajectory; tighter σ_max
should concentrate score-model capacity in the useful regime.

Outcome: not confirmed. On real data (mean over 4 sets) all four EDM variants close
the baseline coverage gap dramatically — I90 abs coverage error drops from 0.068 (B)
to 0.015 (S1), 0.018 (S2), 0.021 (S3), 0.021 (S4). S1 (σ_max=20) ties or wins on every
real-data metric; tighter σ_max does not help. S2 (data-adaptive) is statistically
indistinguishable from S1 and is the safer default if `n` is unknown. On synthetic,
all EDM variants share a small CRPS regression vs B (0.39 vs 0.37) that is independent
of σ_max — a property of the combo, not the schedule.

Recommendation: stop tuning σ_max. Keep σ_max=20 for the winning combo (or
`sde_initialize_from_data=True` as a safer auto-default). Library defaults unchanged.

### 2026-05-13 PF-ODE / Heun sampler sweep (post-fix) — viable sampler, defaults unchanged

Re-run of `pf_ode_sweep.yaml` after the chunked-sampling bug fix. Same config as the
original (2 variants × 8 datasets × 3 seeds × 8 sampler points), now with valid
`n_samples=200` for both Euler-SDE and Heun-PF-ODE rows. See
`results/raw/pf_ode_sweep__all_20260513_115137.jsonl`.

The previous "Heun PF-ODE uniformly worse than Euler" result reverses. With the bug
fixed, Heun PF-ODE on the winning combo improves substantially over the buggy
numbers — for example at n_steps=50 on real data, CRPS goes from 8.62 to **8.21**
and 90% coverage from 0.83 to **0.89**. At matched wall-clock (Heun @ 25 steps ≈
Euler @ 50 steps, both ~0.65s on the winning combo), the two samplers are
effectively tied: Euler edges CRPS by 0.7%, Heun edges coverage by 0.011, intervals
within 3%.

Comparison at matched wall-clock (winning combo, real data):

|                | Euler-SDE @ 50 | Heun-PF-ODE @ 25 |
|----------------|---------------:|-----------------:|
| CRPS           | 8.164          | 8.221            |
| cov@90         | 0.889          | 0.900            |
| `|covE|`@90    | 0.020          | 0.022            |
| width@90       | 46.82          | 48.50            |
| sample time    | 0.65s          | 0.65s            |

Synthetic mirrors this: at n_steps=100, Heun-PF-ODE CRPS=0.371 vs Euler-SDE 0.372 on
the winning combo — tied. Neither sampler wins decisively on any metric of interest.

Findings:
1. **Original negative conclusion was bug-driven, not sampler-driven.** Treat the
   2026-05-12 entry as void; PF-ODE/Heun is a real alternative.
2. **Defaults stay Euler/SDE.** Per-step cost is ~2× lower for Euler, and the
   metric ties don't justify changing the default.
3. **Item #11 (tree-based flow matching) is no longer closed.** It was closed in
   the original write-up on the basis that ODE sampling failed; that premise is
   now wrong. Flow matching returns to "research direction" status — not
   recommended for near-term implementation, but the failure-mode argument no
   longer applies.
4. **The PF-ODE surface is justified, not vestigial.** Keep the sampler available
   for downstream uses (deterministic samples, reproducible quantile estimates,
   future flow-matching prototype).

### 2026-05-12 PF-ODE / Heun sampler sweep — SUPERSEDED (sampling bug)

The original conclusion below was invalidated on 2026-05-13 by the discovery of a
chunked-sampling bug in `_base_tabular_diffusion._sample_with_pipeline`: the prior was
seeded with a constant value across loop iterations, so PF-ODE (which is deterministic
given the prior + score) produced `n_samples / n_parallel` identical copies of
`n_parallel` unique samples. The original sweep used `n_samples=200, n_parallel=20`, so
the effective unique-sample count for PF-ODE rows was 20, not 200. CRPS, coverage, and
width degrade when effective sample size collapses, so the apparent PF-ODE failure was
partly an artifact of the bug rather than a property of the sampler. Bug fixed in
`_base_tabular_diffusion.py` (chunk-indexed seeds for both prior and solver) and
re-run logged below as "PF-ODE / Heun sampler sweep (post-fix)". Original conclusion
preserved for traceability:

> A deterministic probability-flow ODE plus a Heun second-order solver were added behind
> `sampler_method="heun"` + `pf_ode=True`. Step-count sweep at `n_steps ∈ {15, 25, 50, 100}`
> on 8 datasets × 3 seeds, two variants (baseline + winner). See
> `results/raw/pf_ode_sweep__all_20260512_205403.jsonl`. Heun PF-ODE is worse than Euler SDE
> at every step count for both variants: on `residualized_mean_edm_raw_time_log_std`, Euler
> @ 15 steps (CRPS 4.285, I90 err 0.022, time 0.79s) beats Heun PF-ODE @ 100 steps (CRPS
> 4.498, I90 err 0.085, time 6.56s) on every metric while running ~8× faster. The negative
> result was used to close off tree-based flow matching (item #11 from
> `plans/improvements.md`) — that closure is now also provisional pending the re-run.

## Provenance

Every result row records:

- `git_sha`
- `git_dirty`
- `treeffuser_source_hash`

Because `baseline_current` means "the current baseline behavior in this checkout", these
columns are required to interpret old-vs-new comparisons later.
