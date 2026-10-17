"""Synthetic credit-risk model and standalone statistical drift report.
This does NOT send ML monitoring data to Arize AX.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

def dataset(n, shifted=False):
    amount = rng.lognormal(mean=4.0 if not shifted else 4.8, sigma=0.7, size=n)
    account_age = rng.integers(1, 120, n)
    tx_24h = rng.poisson(3 if not shifted else 6, n)
    fraud_logit = -5.5 + 0.012 * amount + 0.30 * tx_24h - 0.008 * account_age
    fraud_prob = 1.0 / (1.0 + np.exp(-np.clip(fraud_logit, -30, 30)))
    fraud = rng.binomial(1, fraud_prob)
    return pd.DataFrame({"amount": amount, "account_age_months": account_age,
                         "transactions_24h": tx_24h, "fraud_label": fraud})

reference, current = dataset(1500), dataset(1500, shifted=True)
columns = ["amount", "account_age_months", "transactions_24h"]
model = LogisticRegression(max_iter=2000)
model.fit(reference[columns], reference.fraud_label)
ref_predictions = model.predict_proba(reference[columns])[:, 1]
cur_predictions = model.predict_proba(current[columns])[:, 1]
features = {}
for col in columns:
    stat, p = ks_2samp(reference[col], current[col])
    features[col] = {"reference_mean": round(float(reference[col].mean()), 3),
                     "current_mean": round(float(current[col].mean()), 3),
                     "ks_statistic": round(float(stat), 4),
                     "p_value": round(float(p), 8), "flagged_p_below_0_05": bool(p < 0.05)}
report = {
    "warning": "Synthetic educational data; unadjusted KS p-values and illustrative model only.",
    "feature_drift": features,
    "predicted_fraud_rate_at_threshold_0_5": {
        "reference": float((ref_predictions > .5).mean()),
        "current": float((cur_predictions > .5).mean())},
    "roc_auc_using_immediately_available_simulated_labels": {
        "reference": float(roc_auc_score(reference.fraud_label, ref_predictions)),
        "current": float(roc_auc_score(current.fraud_label, cur_predictions))},
    "production_note": "Real labels may arrive late; monitor feature and prediction drift before accuracy is measurable."
}
(OUT / "ml_drift_report.json").write_text(json.dumps(report, indent=2))
reference.to_csv(OUT / "reference_predictions.csv", index=False)
current.to_csv(OUT / "current_predictions.csv", index=False)
print(json.dumps(report, indent=2))
print(f"\nWrote reports to {OUT}")
