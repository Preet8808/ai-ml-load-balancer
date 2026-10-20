# Ablations and generalisation

Two questions the main result cannot answer:

1. **Which features carry the model?** The M5 score is one number; dropping a
   feature group and re-scoring says what each group is worth.
2. **Does it work on traffic it has never seen?** Every split in `ml/train.py`
   is by run, but all six profiles appear in both train and test. Holding out a
   whole profile is stricter.

Both use the shipped estimator (XGBoost, same hyperparameters, same seed) and
the same run-level split as `models/training_report_032d9bab9a.json`. The
ablated arms skip cross-validation - the comparison is between arms on one
fixed split, not a search for a better model - and write reports only, never
artifacts, so `models/` is untouched.

## Feature ablations

Command:

```bash
python ml/train.py --output scratch/ablations --only xgboost --no-cv --no-artifacts \
  --tag _no_trend --drop-features cpu_delta_5s,rt_p95_5s_ms
```

| Arm | MAE(log) | MAE(ms) | R²(log) | Spearman | Change in MAE |
|---|---|---|---|---|---|
| Full feature set | 0.5868 | 73.1 | 0.6827 | 0.710 | - |
| No trend (`cpu_delta_5s`, `rt_p95_5s_ms`) | 0.6124 | 78.4 | 0.6565 | 0.686 | +4.4% |
| No request type | 0.6955 | 122.1 | 0.5793 | 0.640 | +18.5% |

The full arm reproduces the M5 report exactly (0.5868 / 73.1 ms), which is the
check that the ablation harness is not a different experiment.

**Request type is the feature group the model cannot lose.** Without the four
one-hot columns the model cannot distinguish a 5 ms `ping` from a 300 ms
`payload` request, and the mixed-cost scenario becomes unlearnable. That matches
the scenario design: the whole point is that request cost differs by endpoint.

**Trend features are worth less than the plan expected.** `rt_p95_5s_ms` is the
single most important feature by gain (0.359), yet dropping it and
`cpu_delta_5s` costs only 4.4% MAE. The other state columns - `rt_ewma_ms`,
`req_rate_5s`, `inflight`, CPU - carry most of the same information at this
traffic scale. Worth stating plainly rather than claiming every planned feature
earned its place.

## Unseen-profile generalisation

Command:

```bash
python ml/generalisation.py --profile burst
python ml/generalisation.py --profile wave
```

Every run of the named profile is removed from training; the model is scored on
that profile alone, against the same naive EWMA baseline.

| Held-out profile | Runs held out | Train rows | Naive MAE(log) | Model MAE(log) | Improvement | R² | Spearman |
|---|---|---|---|---|---|---|---|
| burst | 3 | 142,600 | 1.3677 | 0.5530 | **+59.6%** | 0.6797 | 0.708 |
| wave | 3 | 145,934 | 1.3385 | 0.5521 | **+58.8%** | 0.7013 | 0.737 |

Both are *dynamic* profiles: burst is the one with no consistent live routing
win (the earlier A/B lost; the matrix mean is favourable but not significant),
and wave is a sinusoid the model cannot extrapolate from a single run's shape.
The prediction error barely moves from the in-distribution 0.5868, which says the
model learned server/request cost structure rather than a particular load curve.

Caveat: the held-out profile still uses the same four backends, the same
endpoints and the same CPU quotas. This is generalisation across traffic shape,
not across hardware.

## Live herd ablation

The no-herd-protection arm is a routing rule, not a feature, so it is measured
live: `ml_predictive` scores `predicted_ms` alone and `ml` scores
`predicted_ms * (1 + inflight)`. Both run the burst profile in the same session.
The herd term is faster in 3/3 paired repeats: p95 342 vs 700 ms (median ratio
2.21x), p99 518 vs 1134 ms, 102 vs 73 req/s. Without it, 46-69% of traffic goes
to backend1 and only 1-3% to backend4; with it, the shares are 29/28/29/14. The
full table is in `docs/results.md`; raw output is under
`docs/results/herd_ablation.txt`.

Reproduce:

```bash
python bench/ab_compare.py --profiles burst --policies ml ml_predictive \
  --repeats 3 --duration 60 --drift-threshold 10
```

## Reproducibility

The reports committed with this document:

- `docs/results/ablation_full.json`
- `docs/results/ablation_no_trend.json`
- `docs/results/ablation_no_req_type.json`
- `docs/results/generalisation_burst.json`
- `docs/results/generalisation_wave.json`
