# Paired Wilcoxon signed-rank tests on headline CRPS

Source: `benchmarks/results/raw/paper_real_data_v2_full.jsonl`. Each variant's CRPS is averaged over 3 seeds per dataset; the paired test is run across the ten UCI datasets (n=10). Lower CRPS is better, so a negative signed difference favours the left-hand variant.

## Summary

| Pair | Alt. | n | a-wins | b-wins | ties | median Δ | W | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| score+ vs published | less | 10 | 7 | 3 | 0 | -0.0422 | 10.0 | 0.042 |
| FM vs published | less | 10 | 7 | 3 | 0 | -0.0436 | 11.0 | 0.053 |
| FM vs score+ | two-sided | 10 | 6 | 4 | 0 | -0.0002 | 26.0 | 0.922 |

## Per-dataset mean CRPS (3-seed average)

| dataset | published | score+ | FM |
|---|---|---|---|
| california_housing | 0.2075 | 0.2124 | 0.2143 |
| concrete | 2.2693 | 1.7949 | 1.7730 |
| diabetes | 34.7524 | 34.5151 | 34.4079 |
| energy | 0.2593 | 0.1897 | 0.1881 |
| kin8nm | 0.0756 | 0.0609 | 0.0595 |
| naval | 0.0003 | 0.0002 | 0.0002 |
| power_plant | 1.7151 | 1.5684 | 1.5747 |
| protein | 1.7872 | 1.8472 | 1.8605 |
| wine | 0.3056 | 0.3175 | 0.3172 |
| yacht | 0.3296 | 0.1954 | 0.2089 |

## Per-dataset signed differences

### score+ vs published (alt=less)

| dataset | Δ CRPS |
|---|---:|
| california_housing | +0.0049 |
| concrete | -0.4744 |
| diabetes | -0.2374 |
| energy | -0.0696 |
| kin8nm | -0.0147 |
| naval | -0.0001 |
| power_plant | -0.1466 |
| protein | +0.0600 |
| wine | +0.0119 |
| yacht | -0.1342 |

### FM vs published (alt=less)

| dataset | Δ CRPS |
|---|---:|
| california_housing | +0.0068 |
| concrete | -0.4962 |
| diabetes | -0.3445 |
| energy | -0.0712 |
| kin8nm | -0.0161 |
| naval | -0.0001 |
| power_plant | -0.1404 |
| protein | +0.0732 |
| wine | +0.0116 |
| yacht | -0.1207 |

### FM vs score+ (alt=two-sided)

| dataset | Δ CRPS |
|---|---:|
| california_housing | +0.0019 |
| concrete | -0.0219 |
| diabetes | -0.1071 |
| energy | -0.0016 |
| kin8nm | -0.0014 |
| naval | -0.0000 |
| power_plant | +0.0062 |
| protein | +0.0133 |
| wine | -0.0003 |
| yacht | +0.0135 |
