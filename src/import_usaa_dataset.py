"""Import draft reference answers into Phoenix; this does not run an accuracy test."""
import json
import time
from pathlib import Path
import requests

BASE = 'http://localhost:6006'
NAME = 'usaa-public-information-draft-v1'
cases = json.loads((Path(__file__).resolve().parents[1] / 'data/usaa-evaluation-cases.json').read_text())['cases']
r = requests.get(f'{BASE}/v1/datasets', timeout=10)
r.raise_for_status()
existing = next((d for d in r.json()['data'] if d['name'] == NAME), None)
if existing:
    print('Dataset already exists:', existing)
else:
    payload = {
        'action': 'create', 'name': NAME,
        'description': '46 researcher-drafted USAA public-information references. Not policy contracts; not independently adjudicated. No measured model accuracy. Reference answers must not be passed to the answering model.',
        'inputs': [{'question': c['question'], 'product': c['product']} for c in cases],
        'outputs': [{'reference_answer': c['reference_answer']} for c in cases],
        'metadata': [{k: c[k] for k in ['id', 'source_url', 'source_section', 'source_type', 'access_status', 'reviewed_on', 'review_status', 'must_check']} for c in cases],
        'splits': ['source_unavailable' if c['access_status'] == 'blocked_cookie_screen' else 'draft_reference' for c in cases],
    }
    r = requests.post(f'{BASE}/v1/datasets/upload', json=payload, timeout=30)
    r.raise_for_status()
    print('Imported draft dataset:', r.json())
for attempt in range(20):
    r = requests.get(f'{BASE}/v1/datasets', timeout=10)
    r.raise_for_status()
    dataset = next((d for d in r.json()['data'] if d['name'] == NAME), None)
    if dataset:
        print('Verified:', dataset)
        break
    time.sleep(0.5)
else:
    raise SystemExit('Upload accepted but dataset not visible yet; check Phoenix before retrying.')
