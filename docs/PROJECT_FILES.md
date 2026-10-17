# Project file index

All generated chat deliverables and download files were consolidated here on October 17, 2026. Identical copies were verified with SHA-256 before removing the duplicate from the chat workspace. No differing project file was overwritten. The move log is `.local/organization-manifest.json`.

| Folder/file | Contents |
|---|---|
| `src/` | Runnable GenAI, ML and Phoenix import/evaluation scripts |
| `tests/` | Traditional ML calculation and data-join tests |
| `data/usaa-evaluation-cases.json` | 46 draft questions and reference answers |
| `data/usaa-pilot-contexts.json` | Five curated summaries supplied to Qwen3 |
| `data/insurance/` | Both OpenML ARFF downloads and their provenance/checksum manifest |
| `docs/usaa-research-and-test-plan.md` | Research inventory, citations and evaluation plan |
| `outputs/usaa-local-pilot-results.md` | Readable GenAI pilot findings |
| `outputs/usaa-local-pilot-20261017T223512Z.json` | GenAI answers, local judge scores and assistant reviews |
| `outputs/insurance-ml-20261017T233659Z/` | Final Tweedie model report, verification, predictions, batch and segment metrics |
| `outputs/insurance-ml-20261017T233637Z/` | Earlier exploratory tree-model run, superseded |
| `outputs/ml_drift_report.json` | Original synthetic transaction drift report |
| `outputs/reference_predictions.csv`, `outputs/current_predictions.csv` | Original synthetic transaction data exports |
| `outputs/*progress.log` | Evaluation execution logs |
| `scripts/build_usaa_research.py` | Script that assembled the draft research outputs |
| `.local/installers/` | Ollama ZIP, downloaded install script and extracted app bundle |
| `.local/research/phoenix-openapi.json` | Local Phoenix API schema downloaded during implementation |

## Start here

Read the final ML report and the GenAI pilot report above. See the root README for commands.

Generated outputs, raw ARFF downloads, installer archives and the virtual environment are ignored by Git. They are in this local project folder but are not automatically uploaded to GitHub. Source, tests, research notes, reference JSON and the download manifest can be versioned. No commit or push was made during this organization step.

The running Ollama installation and its model cache remain in their standard macOS locations; Phoenix's database remains in its Docker volume. Those application stores were not relocated. Earlier chat download links now point to moved files; use this project index for their new locations.
