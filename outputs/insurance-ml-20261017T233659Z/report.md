# Traditional ML insurance evaluation

Run: 20261017T233659Z. Phoenix project: `insurance-traditional-ml`.

Model: regularized Tweedie regression (power 1.5, log link, alpha 1.0). It predicts annual cost, converted to policy-period cost using exposure. Train: 120,000 policies; held-out test: 30,000 policies. Claim counts and amounts are not input features.

| Metric | Baseline | Simulated 1.25x costs |
|---|---:|---:|
| MAE (EUR) | 159.51 | 180.01 |
| RMSE (EUR) | 1694.18 | 2117.09 |
| Mean actual cost (EUR) | 82.57 | 103.21 |
| Mean predicted cost (EUR) | 87.44 | 87.44 |
| Mean actual minus prediction (EUR) | -4.87 | 15.77 |
| Predicted / actual total | 1.059 | 0.847 |
| Tweedie deviance | 81.84 | 92.15 |
| Constant-rate baseline MAE (EUR) | 166.10 | 186.66 |

## Interpretation

Inputs, exposure and predictions are identical across the two batches. Only actual claim amounts are multiplied by 1.25. Therefore input drift is zero by construction, while outcome costs and prediction error can change. This illustrates why input drift alone cannot establish model quality. Metrics are calculated from outcomes, without an LLM judge. It does not show a real-world change over time.

A constant-rate comparator is included; this run does not establish that Tweedie regression is the best model. MAE alone favors low predictions in sparse claims, so inspect aggregate calibration and deviance too. Segment metrics are descriptive, not causal or fairness conclusions.

## Data and limitations

- Historical French motor third-party liability data, not USAA data or modern repair estimates.
- Random policy holdout, not a temporal backtest; train and test IDs are disjoint.
- Severity records summed per policy; unmatched severity is audited; absent severity becomes zero.
- Outcome inflation is an intentional simulation, not an observed drift finding.
- Model trained on a reproducible sample to fit laptop resources; no hyperparameter search.
- Trace samples intentionally include large losses and are not representative.
- Metrics computed in Python and displayed as Phoenix CODE annotations; no native Phoenix drift engine or LLM judge.

Source: https://www.openml.org/d/41214 and https://www.openml.org/d/41215; CC0 according to the downloaded metadata. See data/insurance/manifest.json.

## Validation

Four unit tests passed. The saved 60,000 prediction records were independently checked for matching policy IDs, identical predictions and exposure, exact 1.25x outcomes, and matching recalculated MAE. Phoenix stored 46 metric annotations; both batch MAE scores match this report. An additional run-status annotation marks this final run.

The earlier tree-model experiment had a predicted/actual ratio of about 10.61 and was superseded by this regularized Tweedie run. It remains visible in Phoenix with an exploratory/superseded status for transparency.
