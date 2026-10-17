# Enterprise AI Observability — POC v0.1

Demonstrates **two distinct observability categories** using synthetic data:

- `src/ml_drift.py`: traditional ML drift, changing predictions, and illustrative quality metrics. Saves local JSON/CSV reports. **This is NOT connected to Arize AX.**
- `src/llm_traces.py`: deterministic synthetic RAG-like workflow, traced to **Arize Phoenix** using OTLP/HTTP. Simulates correct and incorrect answers, with a simple deterministic check against an approved synthetic policy. **This does NOT call a real LLM or run a Phoenix-native evaluation.**

## Requirements

- Python 3.10+ (3.11 or 3.12 recommended)
- Docker and Docker Compose for Phoenix

## Run

```bash
# from the project directory
docker compose up -d
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/ml_drift.py
python src/llm_traces.py
```

Browse **http://localhost:6006**, select your trace project/service, and inspect spans for `agent.answer`, `retrieve_policy`, `generate_response`, and `evaluate_groundedness`. If the UI isn't ready, give the container time to initialize and rerun the trace script.

ML results are saved under `outputs/ml_drift_report.json` plus two CSV files. View with any text editor or spreadsheet application.

## Suggested 5-minute demo

1. Open `outputs/ml_drift_report.json`. Show a changed feature distribution and predicted fraud rates, emphasizing why monitoring model quality requires more than normal application metrics.
2. Open Phoenix and show trace trees for two synthetic answers: one passes the approved-policy check, one fails.
3. Explain how different model categories need distinct telemetry pipelines but share governance, RBAC, retention, cost allocation, and onboarding concerns.

## Limitations, security, next steps

- Educational, simulated data only. No real bank/customer information is used.
- `docker-compose.yml` binds Phoenix to localhost and **does not configure authentication**. Do not expose on a shared network or deploy as production-is.
- `latest` image and loose dependency bounds speed prototyping; pin and verify versions before formal evaluation.
- KS p-values are unadjusted; statistical drift is not the same as concept drift, causal model degradation, or business impact.
- The local ML report **does not demonstrate Arize AX ML monitoring**. To test AX ingestion, obtain an authorized AX workspace and follow its model/prediction ingestion documentation.
- Phoenix traces use manual OTel spans with simulated behavior; replace with a genuine instrumented LLM call and recorded evaluation datasets in the next iteration.
- This POC does **not** establish AX/Phoenix enterprise equivalence or prove multi-tenant isolation.

## References

- Phoenix self-hosting: https://arize.com/docs/phoenix/self-hosting/deploying-phoenix
- Phoenix tracing endpoint: https://arize.com/docs/phoenix/resources/frequently-asked-questions/what-is-my-phoenix-endpoint/
- OpenTelemetry Python documentation: https://opentelemetry.io/docs/languages/python/
