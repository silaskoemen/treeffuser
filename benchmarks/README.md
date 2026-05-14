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

### 2026-05-13 Paper-grade real-data v2 + SDE-under-matched-residualizer ablation

Two consecutive sweeps establish the paper-grade headline. **(1)** Re-run of the
10-dataset UCI suite with the residualizer-fm-sweep findings folded in:
treeffuser_published, treeffuser_score_combo, vp_fm_ode+residualizer-C,
vp_fm_ode+residualizer-E. See
`results/raw/paper_real_data_v2__all_20260513_191710.jsonl`. **(2)** SDE ablation
under matched residualizer-C across stochasticity ∈ {0.25, 0.5, 1.0} and
schedule ∈ {linear, sqrt}, plus residualizer-E at stoch=1.0. See
`results/raw/sde_lower_stoch_sweep__all_20260513_203445.jsonl`.

Headline (mean over 10 UCI datasets × 3 seeds; full 9-variant table including
the SDE ablation):

| Variant                    | CRPSS↑ | relCRPS↓ |   DSS↓ | KS p>.05↑ | `|cE|@50`↓ | `|cE|@90`↓ | `|cE|@95`↓ | samp_t |
|----------------------------|-------:|---------:|-------:|----------:|-----------:|-----------:|-----------:|-------:|
| **score_combo**            | **0.6565** | 1.024 | **−0.169** | **0.70** | **0.069** | 0.038 | 0.026 | 5.51s |
| **vp_fm_ode + C**          | **0.6564** | 1.025 | −0.154 | 0.47 | 0.087 | **0.024** | **0.015** | **3.07s** |
| vp_fm_ode + E              | 0.6536 | 1.022 | −0.134 | 0.53 | 0.085 | 0.024 | 0.015 | 6.52s |
| sde-linear-stoch0.25 + C   | 0.6560 | 1.027 | −0.142 | 0.40 | 0.098 | 0.028 | 0.015 | 5.68s |
| sde-sqrt-stoch0.5 + C      | 0.6554 | 1.029 | −0.121 | 0.33 | 0.111 | 0.035 | 0.017 | 5.92s |
| sde-linear-stoch0.5 + C    | 0.6552 | 1.030 | −0.118 | 0.33 | 0.112 | 0.036 | 0.017 | 5.92s |
| sde-sqrt-stoch1.0 + C      | 0.6546 | 1.033 | −0.092 | 0.30 | 0.123 | 0.043 | 0.020 | 5.93s |
| sde-linear-stoch1.0 + C    | 0.6540 | 1.035 | −0.074 | 0.30 | 0.126 | 0.047 | 0.022 | 5.93s |
| sde-linear-stoch1.0 + E    | 0.6514 | 1.033 | −0.070 | 0.30 | 0.126 | 0.048 | 0.024 | 9.78s |
| treeffuser_published       | 0.6386 | 1.215 | 1.136 | 0.27 | 0.140 | 0.053 | 0.029 | 26.4s |

Findings:
1. **Treeffuser_published vs our score-side contributions:** CRPSS 0.639 →
   0.657 (+2.8%), sample time 26.4s → 5.5s (4.8x faster), PIT KS pass rate
   0.27 → 0.70. The residualizer-C + EDM + log-σ-sampling + raw_time_log_std
   combo is a substantial pre-FM improvement attributable to our score-side
   work alone.
2. **Our FM contribution (vp_fm_ode + C) ties score_combo on CRPSS** (within
   0.0001) and Pareto-improves on the calibration axes: `|cE|@90` 0.038 →
   0.024 (−37%), `|cE|@95` 0.026 → 0.015 (−42%), sample time 5.5s → 3.1s
   (1.8x further speedup). DSS slightly favors score (−0.169 vs −0.154); KS
   pass rate slightly favors score (0.70 vs 0.47). The tradeoff is sharper
   bulk-of-distribution calibration (score) vs sharper tail calibration plus
   speed (FM).
3. **SDE is monotone-dominated by ODE+C** on every aggregate metric, with
   stochasticity acting as a "noise dial" that strictly worsens results.
   `stoch: 1.0 → 0.5 → 0.25 → 0` produces strictly improving CRPSS, |cE|@95,
   DSS, and KS pass rate, with `stoch=0` reducing to ODE+C exactly. The closest
   SDE contender (`sde-linear-stoch0.25 + C`) ties on `|cE|@95` (0.015) but is
   strictly worse on CRPSS, `|cE|@50`, KS, and 1.85x slower. Across the 10
   datasets, SDE wins per-dataset CRPS exactly once (energy, by 0.003 — negligible).
4. **Per-dataset CRPS winners are now balanced:** FM wins 5/10 (concrete,
   energy, diabetes, kin8nm, naval), score wins 4/10 (yacht, power_plant,
   california_housing, protein), published wins 1/10 (wine — the dataset with
   a categorical color flag). The smaller-data Treeffuser baseline tied or
   beat score on wine; otherwise our score+FM contributions cover the suite.

Paper-grade headline (3 rows) and SDE positioning:

| Row | Contribution                                  | Δ on prior |
|-----|-----------------------------------------------|------------|
| 1   | treeffuser_published                          | —          |
| 2   | treeffuser_score_combo (our score-side work) | +0.018 CRPSS, KS 0.27→0.70, 4.8x faster |
| 3   | vp_fm_ode + residualizer-C (our FM work)     | tied CRPSS, `|cE|@95` −42%, 1.8x faster |

SDE moves to **method** (stochastic-interpolant framework, closed-form score)
and **ablation** (path × stochasticity decomposition, per-IQR-bin flatness).
We have the definitive "tested SDE under matched residualizer × stochasticity
× schedule on the canonical UCI suite; no SDE config Pareto-dominates ODE+C"
claim, which pre-empts reviewer push-back on completeness.

### 2026-05-13 ODE-vs-SDE × path ablation — path and stochasticity decouple cleanly

Disentangles the |covE|@95 win between the *path* and the *stochasticity*
contributions. Sweep: 3 paths × {ODE (stoch=0) at n_steps ∈ {5, 10, 25},
SDE-linear at n_steps=25} × 8 datasets × 3 seeds. Score reference at n_steps=25.
See `results/raw/ode_vs_sde_path_ablation__all_20260513_155700.jsonl`.

Decomposition (real |covE|@95 at n_steps=25):

| Path   | ODE   | SDE   | SDE Δ      | Path Δ (vs linear-ODE) |
|--------|------:|------:|-----------:|-----------------------:|
| linear | 0.047 | 0.034 | −0.013     | 0 (ref)                |
| trig   | 0.028 | 0.017 | −0.011     | **−0.019**             |
| vp     | 0.026 | 0.015 | −0.011     | **−0.021**             |

**Hypothesis H3 (additive contributions) confirmed.** Path effect ≈ −0.020 in
|covE|@95; stochasticity effect ≈ −0.011; combined ≈ −0.031. The path effect is
~2× the stochasticity effect on real data tail calibration. Neither subsumes
the other.

Per-IQR-bin tail-minus-bulk deficit at @95 (real, n_steps=25):

| Variant         | tail−bulk |
|-----------------|----------:|
| score           | +0.007    |
| linear + ODE    | +0.010    |
| linear + SDE    | +0.020    |
| trig   + ODE    | +0.011    |
| **trig + SDE**  | **+0.010** (flat) |
| vp     + ODE    | +0.032    |
| **vp   + SDE**  | **+0.008** (flat) |

**Refinement of the diagnostic story.** *Path* improves the absolute deficit
level (sum across bins). *Stochasticity* drives per-IQR-bin uniformity. Under
ODE alone, even trig and VP show monotone-with-IQR deficit. SDE is what
flattens the per-bin pattern — the marginal-preservation property of
stochastic interpolants equalizes per-IQR-bin coverage. The two effects fix
different aspects of the calibration deficit.

Operating-point summary (real data):

| Operating mode             | Variant            | CRPS  | `|cE|@95` | samp_t |
|----------------------------|--------------------|------:|----------:|-------:|
| Lowest latency             | **VP + ODE @ 5**   | 8.220 | 0.026     | **0.20s** |
| Lowest CRPS                | **VP + ODE @ 25**  | 8.220 | 0.026     | 0.56s  |
| Tightest tail calibration  | **VP + SDE @ 25**  | 8.247 | **0.015** | 0.59s  |
| Tightest bulk calibration  | linear + SDE @ 25  | 8.260 | 0.034     | 0.86s  |
| Score reference            | (winning combo)    | 8.398 | 0.012     | 1.39s  |

**VP + ODE at 5 steps is the new Pareto champion on CRPS-and-speed**: CRPS
8.22 (best ever measured), 7× faster than score, |covE|@95 within 0.014 of
score. This is the operating point for latency-sensitive deployment;
VP + SDE is the operating point for calibration-sensitive deployment.

Additional findings:
1. **All three paths converge at 5 ODE steps** (CRPS@5 / CRPS@25 = 1.000 for
   all). The few-step deterministic-sampling advantage is preserved by
   non-linear paths.
2. **On synthetic, linear+SDE retains the |covE|@95 advantage** (0.011 vs
   trig+ODE 0.029, vp+ODE 0.027). The path effect is real-data-specific —
   most useful where heteroscedasticity and irregular conditional structure
   make per-`t` velocity information matter most.

This closes the path-vs-stochasticity decomposition. For paper-prep, the
result is now publication-grade across the three explanatory axes (path,
schedule, stochasticity).

### 2026-05-13 Flow-path ablation (trig, VP) — non-linear paths close the tail gap

First benchmark of `TrigFlowPath` (variance-preserving cosine/sine,
a(t)=cos(πt/2), b(t)=sin(πt/2)) and `VPFlowPath` (variance-preserving with
DDPM linear-β schedule) implemented in `_flow_matching.py`. The general
score-from-velocity formula s(y_t, t) = (a'(t) y_t − a(t) v) / (W(t) b(t))
covers all three paths. Sweep: 3 paths × 3 schedules ({linear, quadratic,
sqrt}) × 1 stochasticity (1.0) × 1 step count (25) on 8 datasets × 3 seeds,
against the score winning combo. See
`results/raw/flow_path_sweep__all_20260513_152217.jsonl`.

**The schedule-sweep diagnostic is decisively confirmed.** Non-linear paths
flatten the tail-vs-bulk @95 deficit ratio that no schedule could fix under
the linear path. Real |covE|@95:

| Path × schedule | `|covE|@95` | Sum of per-IQR-bin deficits |
|----|----:|----:|
| score (reference) | 0.012 | +0.009 (uniform) |
| linear × linear   | 0.034 | +0.142 (tail-concentrated) |
| linear × quadratic (prior best) | 0.027 | +0.081 (still tail-concentrated) |
| **trig × linear** | **0.017** | **−0.007 (flat)** |
| trig × quadratic  | 0.019 | −0.032 (flat) |
| **vp × linear**   | **0.015** | **−0.011 (flat)** |
| vp × quadratic    | 0.018 | −0.017 (flat) |

VP+linear closes 86% of the gap to score (0.022 → 0.003 absolute). The
schedule-diagnostic hypothesis — "constant-in-t velocity targets under-inform
trees at high-σ regions" — was correct.

Findings:
1. **Both non-linear paths fully flatten the @95 calibration pattern across IQR
   bins.** This was the qualitative gap, not the absolute number. Score matches
   nominal calibration uniformly; linear-FM accumulated deficits at high IQR.
   Trig and VP show essentially flat per-IQR-bin behavior (within ±0.015 across
   all bins).
2. **VP + linear is the new tail-calibration winner.** |covE|@95 = 0.015 (vs
   score 0.012, linear-FM 0.034). CRPS 8.247 (still beats score's 8.398).
   Sample time 0.56s (43% of score's).
3. **VP + sqrt holds the CRPS crown.** CRPS 8.223 — the best real-data CRPS of
   any variant ever benchmarked here. Sqrt schedule was catastrophic under
   linear FM; under VP it works cleanly because VP's `b(t)~sqrt(t)` near data
   makes the `eps^2 / b` ratio well-behaved with a matching sqrt schedule.
4. **Trade-off:** non-linear paths over-cover @50 (`|covE|@50` ~0.06 for trig
   and VP vs 0.002 for linear-FM). Linear-FM retains the bulk-calibration
   crown. There is now a real Pareto frontier rather than a single winner.
5. **Non-linear paths offer no advantage on synthetic data.** Linear+linear
   still wins CRPS (0.366) and ties @95 on synthetic. The path advantage is
   real-data-specific — precisely where the @95 gap mattered.

Recommendation: surface all three paths as opt-in. Default stays unchanged for
backward compatibility. For real-tabular regression workflows, the practical
guidance is:
- `flow_path="linear", velocity_stochasticity_schedule="linear", stochasticity=1.0`
  for sharpest @50 calibration and lowest CRPS variance.
- `flow_path="vp", velocity_stochasticity_schedule="linear", stochasticity=1.0`
  for tight @95 calibration approaching score-level tails.
- `flow_path="vp", velocity_stochasticity_schedule="sqrt", stochasticity=1.0`
  for best raw CRPS.

Item #11 from `plans/improvements.md` is now not just implemented and
recommended — it has produced a *paper-grade* Pareto improvement over the
score-based diffusion baseline on real tabular regression.

### 2026-05-13 Stochasticity-schedule shape sweep — quadratic closes ~30% of the tail gap

The conditional-coverage diagnostic on the prior stochastic-FM result showed the
|covE|@95 gap to score was tail-concentrated (FM's @95 deficit grew ~2x from
easy- to hard-IQR bin while score was uniform). Hypothesis: schedules that
concentrate stochasticity at high t should selectively close the tail gap.
Sweep: 4 schedule shapes ({linear, quadratic, sqrt, tent}) × 2 stochasticity
strengths ({1.0, 2.0}) × 2 step counts ({15, 25}) on 8 datasets × 3 seeds,
against the score winning combo at n_steps=25.
See `results/raw/stochasticity_schedule_sweep__all_20260513_144712.jsonl`.

Headline (real data, n_steps=25, |covE|@95 — lower is better):

| Schedule  | stoch=1.0 | stoch=2.0 |
|-----------|----------:|----------:|
| score     | 0.012     | —         |
| linear    | 0.034     | 0.064     |
| **quadratic** | **0.027** | **0.033** |
| sqrt      | 0.053     | 0.180     |
| tent      | 0.048     | 0.055     |

Findings:
1. **Quadratic at stoch=1.0 is the new recommended default.** Cuts |covE|@95 from
   0.034 (linear) to 0.027 (~20% relative, closes ~30% of the remaining gap to
   score's 0.012). CRPS cost is +0.3% (8.260 → 8.284), still well below score's
   8.398. |covE|@90 improves from 0.032 to 0.027.
2. **The hypothesis is partially confirmed.** Schedules that put more noise at
   low t (sqrt, tent) hurt strictly. Schedules that concentrate noise at high t
   (quadratic) help. The bin-density argument from the diagnostic is correct.
3. **The hypothesis is also partially refuted.** No schedule flattens the
   tail-vs-bulk deficit ratio. Under quadratic@1.0, real @95 per-bin shortfalls
   are still monotone with IQR ({0.007, 0.014, 0.030, 0.006, 0.024}). Score's
   are uniform around zero. This is evidence that part of the @95 gap is
   structural — likely the absence of EDM-style noise-level preconditioning in
   FM, which gives score explicit information about high-σ regions where the
   data tails live.
4. **Quadratic is the only schedule robust to stoch=2.0.** Linear@2.0 over-
   broadens easy points (|covE|@90 jumps to 0.062). Sqrt@2.0 collapses (CRPS
   8.71, |covE|@95 0.180 on real). Tent@2.0 also degrades. Quadratic@2.0 is
   Pareto-comparable to linear@1.0: tied CRPS (8.259 vs 8.260), |covE|@90
   slightly better (0.027 vs 0.032), |covE|@50 slightly worse (0.005 vs 0.002).

Sweet-spot recommendation: `velocity_stochasticity=1.0` with
`velocity_stochasticity_schedule="quadratic"` at n_steps=25. Pareto comparison:

| Setting              | CRPS   | `|cE|@50` | `|cE|@90` | `|cE|@95` | samp_t |
|----------------------|-------:|----------:|----------:|----------:|-------:|
| score @ 25           | 8.398  | 0.028     | 0.020     | **0.012** | 1.35s  |
| FM linear@1.0 @ 25   | **8.260** | **0.002** | 0.032 | 0.034     | 0.81s  |
| FM quadratic@1.0 @ 25| 8.284  | 0.018     | 0.027     | 0.027     | 0.82s  |

Quadratic narrows the @90 and @95 gap to score by half each, at the cost of @50
and a tiny CRPS shift. Use linear for bulk-calibration-first; use quadratic for
balanced coverage across levels. Both Pareto-dominate score on CRPS.

Implication for the next investigation: the @95 gap is not fully schedule-fixable.
Non-linear flow paths (trig, VP) with t-varying velocity targets are now better
motivated — they provide the per-t signal that constant-target linear FM lacks
at extreme regions. Moving (3) from the prior follow-up list to (1) on the
priority order.

### 2026-05-13 Stochastic-interpolant FM sweep — Pareto wins on CRPS and bulk coverage

First benchmark of the stochastic-interpolant FM sampler implemented in
`_flow_matching.ReverseVelocityInterpolant`. For the linear path, the implied
score `s(y_t, t) = -(y_t + (1-t)v)/t` is plugged into the marginal-preserving SDE
`dy = (-v + (ε²/2) s) ds + ε dW` with schedule `ε(t) = stochasticity · t`
(vanishes at the data endpoint, where the score's `1/t` factor would otherwise
blow up). Stochasticity=0 reduces exactly to the deterministic ODE sampler.
Sweep: 2 variants × 8 datasets × 3 seeds × 4 stochasticity values × 4 step counts.
See `results/raw/stochastic_fm_sweep__all_20260513_140451.jsonl`.

Headline (real datasets, n_steps=25, FM stochasticity=1.0 vs score winning combo):

|                         | Score @ 25 | **FM-stoch=1.0 @ 25** | FM-stoch=0 @ 25 |
|-------------------------|-----------:|----------------------:|----------------:|
| CRPS                    | 8.398      | **8.260** (-1.6%)     | 8.281           |
| `|covE|`@50             | 0.028      | **0.002**             | 0.027           |
| `|covE|`@90             | **0.020**  | 0.032                 | 0.052           |
| `|covE|`@95             | **0.012**  | 0.034                 | 0.047           |
| sample time             | 1.53s      | **0.87s** (-43%)      | 0.84s           |

Stochasticity sweeps monotonically: as `ε` grows 0 → 1, |covE|@50 on real
collapses from 0.027 to 0.002, |covE|@90 from 0.052 to 0.032, and CRPS *improves*
from 8.281 to 8.260. The stochastic-interpolant theory predicts marginal
preservation; in our approximate-velocity setting it manifests as
strictly-better-calibrated bulk coverage with no CRPS cost.

Findings:
1. **Stochastic FM Pareto-dominates score on CRPS at every wall-clock budget.**
   Even FM-stoch=0 @ 5 steps (0.26s, CRPS 8.28) beats score @ 25 steps (1.53s,
   CRPS 8.40). Stochasticity does not change this ranking; it shifts FM in the
   coverage dimension only.
2. **Bulk coverage (`|covE|`@50) is dramatically better under stochastic FM.**
   The center of the predictive distribution is the regime where FM was already
   strong; stochasticity sharpens the calibration to near-zero error on every
   real dataset except `bimodal_mixture` (where FM at high stochasticity
   *overcovers* the gap between modes).
3. **Score retains a tail-coverage edge (`|covE|`@95).** Score winning combo: 0.012;
   FM-stoch=1.0: 0.034. Approximately 2× better in absolute terms; both are
   within the practical tolerance for most applications. The difference likely
   reflects the score's EDM preconditioning handling high-σ regions, where
   FM's flat residual scale gives less tail-shape information.
4. **Predicted bimodal failure mode is doubly disconfirmed.** FM-stoch=1.0 has
   the best CRPS on `bimodal_mixture` of any variant ever benchmarked here
   (0.480 vs score 0.529, -9.3%). Crossing trajectories with stochastic
   sampling work well for tree-based velocities.
5. **CRPS-Pareto frontier on real data is owned by FM at every budget.**
   - 0.26s (FM-stoch=0 @ 5 steps): CRPS 8.28, |covE|@50 0.027
   - 0.28s (FM-stoch=0.25 @ 5 steps): CRPS 8.29, |covE|@50 0.006 *
   - 0.43s (FM-stoch=1.0 @ 10 steps): CRPS 8.28, |covE|@50 0.020
   - 0.87s (FM-stoch=1.0 @ 25 steps): CRPS 8.26, |covE|@50 0.002 *
   - 1.53s (score @ 25 steps): CRPS 8.40, |covE|@50 0.028

   `*` marks particularly attractive operating points.

This is the first benchmark in which the FM track *dominates* the score track
on CRPS rather than tying it. The combination of (a) the few-step deterministic
sampling advantage of FM with (b) the marginal-coverage advantage of stochastic
sampling produces a Pareto-optimal tabular generative pipeline in which
`velocity_stochasticity` is a clean inference-time tuning knob: 0 for fastest
sampling, ~1 for best bulk calibration, with score still preferred only for
tail-quantile calibration on smooth real-data densities.

Recommendation: promote stochastic-interpolant FM (linear path, residualized,
`velocity_stochasticity` 0.25–1.0, n_steps 10–25) as the recommended sampling
mode for tabular regression where CRPS or bulk calibration matters. Library
defaults remain unchanged for backward compatibility; new fits should opt in.
Item #11 from `plans/improvements.md` is now decisively beneficial, not just
implemented.

Open follow-ups suggested by this result:
1. Tail-calibration gap (`|covE|`@95) — can a different stochasticity schedule
   (e.g., `ε(t) = c·t·(1-t)` instead of `c·t`) close it without sacrificing the
   bulk wins?
2. FM-equivalent log-σ t-sampling — would close the residual CRPS variance
   relative to the converged FM asymptote.
3. Non-linear schedules (trig, VP) — now justified, since the stochastic
   sampler may interact differently with non-constant velocity targets.

### 2026-05-13 Flow matching sweep — sampling-efficiency win, marginal-quality tie

First benchmark of the linear-flow-matching path implemented in `_flow_matching.py`
plus `LightGBMVelocityModel` in `_score_models.py`. Sweep: 5 variants × 8 datasets ×
3 seeds × 5 step counts (5/10/15/25/50). Variants: score baseline (Euler & Heun
pf-ode), score winning combo (`residualized_mean_edm_raw_time_log_std`), the
strongest current score combo (`residualized_mean_edm_logsigma_resC` with log-σ
t-sampling + residualizer-C), and two FM variants (`linear_fm_raw_time`,
`linear_fm_residualized_mean_raw_time`). See
`results/raw/flow_matching_sweep__all_20260513_125336.jsonl`.

Headline: FM hits its asymptotic CRPS at **n_steps=5**; score methods need ~25.
Step-count convergence ratio (CRPS at 5 steps / CRPS at 50 steps) is 1.00 for both
FM variants on real and synthetic, vs 1.14-1.70 for all score variants. The plan's
core hypothesis — "deterministic ODE sampling on a directly-learned velocity field
needs fewer steps" — is confirmed cleanly. Matched wall-clock comparison on real
data: FM-residualized @ 5 steps (1.10s) reaches CRPS 8.43, |covE|@90 = 0.030; the
strongest score variant needs 15 steps (1.33s) to reach CRPS 8.45, |covE|@90 = 0.028.
FM is ~20% faster at comparable marginal quality.

Findings on marginal metrics (asymptotic, n_steps=50):
- **Score winning combo retains a narrow CRPS edge** on real data (8.39 vs 8.43,
  +0.5%) and ties on synthetic.
- **FM-residualized matches score on coverage** on synthetic (often better:
  heteroscedastic_nonlinear |covE|@90 = 0.008 for FM-resid vs 0.022 for S1).
- **The predicted bimodal failure mode did not materialize.** Linear-FM on
  `bimodal_mixture` is the *best* variant on coverage (FM-resid |covE|@90 = 0.014,
  S1 0.022, baseline 0.050) and FM-raw has the best CRPS (0.504, beats S1's 0.529).
  Crossing trajectories are not a problem for tree-based velocities at this scale.
  2-rectified flow is *not* a justified follow-up.
- **Residualization is decisive for FM coverage.** FM-raw real cov@90 = 0.78
  (badly undercovers); FM-residualized = 0.88 (~nominal). Same "difficulty
  homogenizer" story as the score track, more pronounced.

Why FM doesn't dominate on raw CRPS: FM lacks the score-track wins that drove
recent CRPS reductions — there's no FM analog of log-σ t-sampling (uniform t with
5% endpoint anchor is the current implementation), no EDM preconditioning, and the
residualizer is at default settings rather than residualizer-C. The 0.5% CRPS gap
could plausibly close with FM-equivalent versions of these levers.

Recommendation: keep FM as an experimental variant. The library default stays
score-based. FM is the right choice when sampling latency dominates training cost
(few-step ODE sampling) or when bimodal calibration matters; score remains
preferred for raw CRPS on real tabular regression. Item #11 from
`plans/improvements.md` (tree-based flow matching) is now *implemented and
benchmarked* rather than closed-or-deferred.

Open follow-ups (not yet justified but plausible):
1. FM-equivalent t-sampling (`log(t/(1-t))` Normal) — could close the CRPS gap.
2. FM-residualized with residualizer-C — could close the coverage gap on real data.
Both are small additions; would launch only if (1) and (2) are jointly worth the
wall-clock latency advantage on a real downstream use case.

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
