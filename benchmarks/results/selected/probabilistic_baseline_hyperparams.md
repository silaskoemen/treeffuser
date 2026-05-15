# Probabilistic Baseline Hyperparameter Selection

Selection suite: 4 synthetic + 4 small real datasets, 3 seeds.
Aggregation: mean over seeds within each dataset, then unweighted mean over datasets.
Primary metric: `crps` (min).
Tie-breaker within 0.25%: `interval_90_abs_coverage_error`.

## Winners

| Family | Variant | Score | Tie-breaker | Rows |
|---|---|---:|---:|---:|
| card | `card__h100__lr0p001__ep150__layers1__steps50` | 5.68321 | 0.0495694 | 24 |
| catboost_uncertainty | `catboost_uncertainty__d4__it3000__lr0p05` | 4.21683 | 0.0555319 | 24 |
| deep_ensemble | `deep_ensemble__h50__lr0p001__ep150__ens5__layers1` | 4.13666 | 0.0335145 | 24 |
| ibug | `ibug__k100__leaf_sample_trees64__lr0p05__md3__est300` | 4.34275 | 0.0200518 | 24 |
| ngboost | `ngboost__lr0p02__est3000` | 4.43868 | 0.0864486 | 24 |
| qreg_lightgbm | `qreg_lightgbm__lr0p1__est500__leaves31__q31` | 4.27273 | 0.0746027 | 24 |

## Full Ranking

### card

| Rank | Variant | Score | Tie-breaker | Fit time | Rows |
|---:|---|---:|---:|---:|---:|
| 1 | `card__h100__lr0p001__ep150__layers1__steps50` | 5.68321 | 0.0495694 | 0.225 | 24 |
| 2 | `card__h100__lr0p001__ep150__layers1__steps100` | 5.84801 | 0.06225 | 0.229 | 24 |
| 3 | `card__h50__lr0p001__ep150__layers1__steps50` | 5.87697 | 0.0487881 | 0.261 | 24 |
| 4 | `card__h50__lr0p001__ep150__layers1__steps100` | 6.45308 | 0.062625 | 0.237 | 24 |

### catboost_uncertainty

| Rank | Variant | Score | Tie-breaker | Fit time | Rows |
|---:|---|---:|---:|---:|---:|
| 1 | `catboost_uncertainty__d4__it3000__lr0p05` | 4.21683 | 0.0555319 | 0.057 | 24 |
| 2 | `catboost_uncertainty__d4__it3000__lr0p03` | 4.23437 | 0.0596823 | 0.087 | 24 |
| 3 | `catboost_uncertainty__d4__it3000__lr0p1` | 4.23569 | 0.0594892 | 0.038 | 24 |
| 4 | `catboost_uncertainty__d6__it3000__lr0p05` | 4.24982 | 0.0510777 | 0.080 | 24 |
| 5 | `catboost_uncertainty__d6__it3000__lr0p03` | 4.27077 | 0.0582948 | 0.116 | 24 |
| 6 | `catboost_uncertainty__d8__it3000__lr0p03` | 4.32346 | 0.0575915 | 0.211 | 24 |
| 7 | `catboost_uncertainty__d6__it3000__lr0p1` | 4.32413 | 0.0629304 | 0.057 | 24 |
| 8 | `catboost_uncertainty__d8__it3000__lr0p05` | 4.32525 | 0.0612825 | 0.160 | 24 |
| 9 | `catboost_uncertainty__d8__it3000__lr0p1` | 4.32834 | 0.0575847 | 0.108 | 24 |

### deep_ensemble

| Rank | Variant | Score | Tie-breaker | Fit time | Rows |
|---:|---|---:|---:|---:|---:|
| 1 | `deep_ensemble__h50__lr0p001__ep150__ens5__layers1` | 4.13666 | 0.0335145 | 1.295 | 24 |
| 2 | `deep_ensemble__h50__lr0p001__ep150__ens3__layers1` | 4.1895 | 0.0329937 | 0.814 | 24 |
| 3 | `deep_ensemble__h100__lr0p001__ep150__ens5__layers1` | 4.19825 | 0.0411152 | 1.014 | 24 |
| 4 | `deep_ensemble__h50__lr0p001__ep150__ens5__layers2` | 4.20071 | 0.0394869 | 1.339 | 24 |
| 5 | `deep_ensemble__h100__lr0p001__ep150__ens5__layers2` | 4.20901 | 0.0441727 | 1.019 | 24 |
| 6 | `deep_ensemble__h50__lr0p001__ep150__ens3__layers2` | 4.21331 | 0.0420894 | 0.799 | 24 |
| 7 | `deep_ensemble__h100__lr0p001__ep150__ens3__layers1` | 4.21811 | 0.0387672 | 0.684 | 24 |
| 8 | `deep_ensemble__h100__lr0p001__ep150__ens3__layers2` | 4.24417 | 0.0450651 | 0.757 | 24 |

### ibug

| Rank | Variant | Score | Tie-breaker | Fit time | Rows |
|---:|---|---:|---:|---:|---:|
| 1 | `ibug__k50__leaf_sample_trees64__lr0p05__md3__est300` | 4.33877 | 0.0232473 | 0.343 | 24 |
| 2 | `ibug__k50__leaf_sample_trees64__lr0p05__md3__est1000` | 4.33877 | 0.0232473 | 1.110 | 24 |
| 3 | `ibug__k100__leaf_sample_trees64__lr0p05__md3__est300` | 4.34275 | 0.0200518 | 0.339 | 24 |
| 4 | `ibug__k100__leaf_sample_trees64__lr0p05__md3__est1000` | 4.34275 | 0.0200518 | 1.113 | 24 |
| 5 | `ibug__k100__leaf_sample_trees64__lr0p05__md6__est300` | 4.37816 | 0.0195293 | 0.612 | 24 |
| 6 | `ibug__k100__leaf_sample_trees64__lr0p05__md6__est1000` | 4.37816 | 0.0195293 | 1.935 | 24 |
| 7 | `ibug__k50__leaf_sample_trees64__lr0p05__md6__est300` | 4.39016 | 0.0348003 | 0.619 | 24 |
| 8 | `ibug__k50__leaf_sample_trees64__lr0p05__md6__est1000` | 4.39016 | 0.0348003 | 1.890 | 24 |
| 9 | `ibug__k200__leaf_sample_trees64__lr0p05__md3__est300` | 4.8021 | 0.0267874 | 0.350 | 24 |
| 10 | `ibug__k200__leaf_sample_trees64__lr0p05__md3__est1000` | 4.8021 | 0.0267874 | 1.130 | 24 |
| 11 | `ibug__k200__leaf_sample_trees64__lr0p05__md6__est300` | 4.95132 | 0.0246798 | 0.633 | 24 |
| 12 | `ibug__k200__leaf_sample_trees64__lr0p05__md6__est1000` | 4.95132 | 0.0246798 | 1.918 | 24 |

### ngboost

| Rank | Variant | Score | Tie-breaker | Fit time | Rows |
|---:|---|---:|---:|---:|---:|
| 1 | `ngboost__lr0p02__est3000` | 4.43868 | 0.0864486 | 0.533 | 24 |
| 2 | `ngboost__lr0p02__est1000` | 4.44625 | 0.0864781 | 0.554 | 24 |
| 3 | `ngboost__lr0p05__est3000` | 4.58698 | 0.152167 | 0.296 | 24 |
| 4 | `ngboost__lr0p05__est1000` | 4.58698 | 0.152167 | 0.304 | 24 |
| 5 | `ngboost__lr0p1__est1000` | 4.9193 | 0.23547 | 0.213 | 24 |
| 6 | `ngboost__lr0p1__est3000` | 4.9193 | 0.23547 | 0.220 | 24 |

### qreg_lightgbm

| Rank | Variant | Score | Tie-breaker | Fit time | Rows |
|---:|---|---:|---:|---:|---:|
| 1 | `qreg_lightgbm__lr0p1__est500__leaves31__q31` | 4.27273 | 0.0746027 | 12.630 | 24 |
| 2 | `qreg_lightgbm__lr0p05__est500__leaves31__q31` | 4.27975 | 0.0780055 | 18.741 | 24 |
| 3 | `qreg_lightgbm__lr0p05__est500__leaves31__q21` | 4.28386 | 0.0888249 | 12.888 | 24 |
| 4 | `qreg_lightgbm__lr0p1__est500__leaves31__q21` | 4.29652 | 0.0915628 | 8.758 | 24 |
