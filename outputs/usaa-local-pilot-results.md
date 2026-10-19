# USAA local-model pilot results

October 17, 2026. Model: Qwen3 4B through Ollama on the M1 GPU. Phoenix project: `usaa-local-pilot`.

Five generated answers were evaluated against draft reference answers and short curated paraphrases of public USAA pages. References were withheld from answer generation. This was not an automatic retrieval test or a review of complete insurance contracts.

| Case | Local judge | Assistant source review |
|---|---|---|
| Auto collision | Pass | Core answer supported |
| Renters flood protection | Pass | Location-dependent condition preserved |
| Homeowners flood exclusion | Pass | Core answer supported |
| Dental implant information | Pass | Needs revision: invented explicit source statement |
| Pet wellness plan | Pass | Correctly distinguishes wellness from insurance |

Two deliberately wrong fixture answers were rejected by the judge's correctness metric. Both nevertheless received passing condition scores, showing inconsistent judging. Fixtures are excluded from generated-answer statistics.

## Dental finding

The answer appropriately said the overview does not establish personal implant coverage. It then incorrectly claimed USAA explicitly states that the overview provides no implant benefit. This phrase came from a curated research note describing missing information, not from an actual USAA statement excluding coverage. The judge accepted that attribution. This exposes a problem with both source-note presentation and judging.

## Interpretation

The model judge passed 5/5 answers, but this is not a verified accuracy estimate. Assistant source review flagged one answer and two control-score inconsistencies. Neither review is human expert adjudication. The sample is small, selected, and uses the same model for answers and judging. An exact-URL check verifies citation presence only, not whether the citation supports each claim.

## Verified Phoenix storage

17 spans; 28 annotations including 7 assistant review notes. Review notes are marked LLM, not HUMAN. Run ID: `20261017T223512Z`.

## Next changes

Separate original source text from researcher notes. Disallow quoting paraphrases as literal source language. Score unsupported attribution separately. Calibrate the condition judge using labeled correct, incorrect and incomplete examples, and add independent human review before expanding the benchmark.
