"""Deterministic retrieval-and-answer agent instrumented with OTel for Phoenix.
No real language model is called. The behavior is designed to be replayable.
"""
import os
import time
from urllib.request import urlopen
from urllib.error import URLError
from urllib.parse import urlsplit
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:6006/v1/traces")
# Fail before generating spans if the configured collector is not ready.
collector = urlsplit(endpoint)
health_url = f"{collector.scheme}://{collector.netloc}/healthz"
try:
    with urlopen(health_url, timeout=10) as health:
        if health.status != 200:
            raise SystemExit(f"Phoenix is not ready: {health_url}")
except (URLError, OSError) as exc:
    raise SystemExit(f"Phoenix is not ready at {health_url}. Start Docker Compose and retry. ({exc})") from exc

provider = TracerProvider(resource=Resource.create({"service.name": "synthetic-policy-assistant", "service.version": "0.1"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, timeout=10)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("ai-observability-poc")

approved_policies = {"wire_limit": "Daily wire transfer limit is $5,000."}
outdated_policies = {"wire_limit": "Daily wire transfer limit is $10,000."}


def answer_question(question, use_outdated_policy=False):
    with tracer.start_as_current_span("agent.answer") as root:
        root.set_attribute("openinference.span.kind", "CHAIN")
        root.set_attribute("model.type", "deterministic_simulation")
        root.set_attribute("input.value", question)
        with tracer.start_as_current_span("retrieve_policy") as span:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            policy = (outdated_policies if use_outdated_policy else approved_policies)["wire_limit"]
            span.set_attribute("retrieval.source", "outdated" if use_outdated_policy else "approved")
            span.set_attribute("retrieval.policy_text", policy)  # Synthetic text only; don't do this with real PII.
            time.sleep(.05)
        with tracer.start_as_current_span("generate_response") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            response = policy
            span.set_attribute("generation.mode", "simulated_no_llm_api")
            time.sleep(.03)
        with tracer.start_as_current_span("evaluate_groundedness") as span:
            span.set_attribute("openinference.span.kind", "EVALUATOR")
            grounded_against_approved = response == approved_policies["wire_limit"]
            span.set_attribute("evaluation.name", "approved_policy_match")
            span.set_attribute("evaluation.passed", grounded_against_approved)
            span.set_attribute("evaluation.score", 1.0 if grounded_against_approved else 0.0)
        root.set_attribute("output.value", response)
        root.set_attribute("evaluation.passed", grounded_against_approved)
        print(f"Source={'outdated' if use_outdated_policy else 'approved'} | Answer={response} | approved_policy_match={grounded_against_approved}")

for faulty in [True, False, True, False]:
    answer_question("What is the daily wire transfer limit?", use_outdated_policy=faulty)
flushed = provider.force_flush(timeout_millis=10000)
provider.shutdown()
if not flushed:
    raise SystemExit("Trace processor flush timed out; check Phoenix and exporter logs.")
print("Trace export attempted to", endpoint, "— verify ingestion in Phoenix (http://localhost:6006).")
