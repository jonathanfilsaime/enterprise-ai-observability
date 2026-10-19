# Traditional ML insurance evaluation

Run: 20261017T233637Z. Phoenix project: `insurance-traditional-ml`.

Model: histogram gradient boosting with Poisson loss. It predicts annual cost, converted to policy-period cost using exposure. Train: 120,000 policies; held-out test: 30,000 policies. Claim counts and amounts are not input features.

| Metric | Baseline | Simulated 1.25x costs |
|---|---:|---:|
| MAE (EUR) | 948.67 | 969.09 |
| RMSE (EUR) | 27847.65 | 27876.25 |
| Mean actual cost (EUR) | 82.57 | 103.21 |
| Mean predicted cost (EUR) | 876.17 | 876.17 |
| Mean actual minus prediction (EUR) | -793.61 | -772.96 |
| Predicted / actual total | 10.611 | 8.489 |
| Tweedie deviance | 88.43 | 99.13 |
| Constant-rate baseline MAE (EUR) | 166.10 | 186.66 |

## Interpretation

Inputs, exposure and predictions are identical across the two batches. Only actual claim amounts are multiplied by 1.25. Therefore input drift is zero by construction, while outcome costs and prediction error can change. This illustrates why input drift alone cannot establish model quality. Metrics are calculated from outcomes, without an LLM judge. It does not show a real-world change over time.

A constant-rate comparator is included; this run does not establish that gradient boosting is the best model. MAE alone favors low predictions in sparse claims, so inspect aggregate calibration and deviance too. Segment metrics are descriptive, not causal or fairness conclusions.

## Data and limitations

- Historical French motor third-party liability data, not USAA data or modern repair estimates.
- Random policy holdout, not a temporal backtest; train and test IDs are disjoint.
- Severity records summed per policy; unmatched severity is audited; absent severity becomes zero.
- Outcome inflation is an intentional simulation, not an observed drift finding.
- Model trained on a reproducible sample to fit laptop resources; no hyperparameter search.
- Trace samples intentionally include large losses and are not representative.
- Metrics computed in Python and displayed as Phoenix CODE annotations; no native Phoenix drift engine or LLM judge.

Source: https://www.openml.org/d/41214 and https://www.openml.org/d/41215; CC0 according to the downloaded metadata. See data/insurance/manifest.json.
