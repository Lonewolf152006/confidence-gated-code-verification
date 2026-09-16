# Usage-Semantic Hallucination Detection in LLM-Generated Code

**Working title**: *"Confidence-Gated Detection of Usage-Semantic Hallucinations in LLM-Generated Code: A Lightweight Alternative to LLM-Based Verification"*

## Problem

LLMs writing code sometimes use a real, syntactically valid API — but the wrong one for the task ("Intent Misuse"). This is common (20–30% of method-level misuses) and static analysis tools cannot detect it, since nothing is syntactically incorrect.

## Architecture: Confidence-Gated Verification Cascade

1. **Layer 0+1 (cheap)**: static existence check + token-level entropy/self-consistency signals — near-free
2. **Router**: confidence-gated decision — most cases accepted at low cost
3. **Escalation**: uncertain cases → small distilled specialist model
4. **Output**: labeled result with misuse category + confidence

## Project Structure

```
ai_project/
├── src/
│   ├── parse_calls.py          # AST-based API call extraction + def-use chains
│   ├── baselines.py            # pylint/mypy baseline runners
│   ├── classifier.py           # GroupKFold classifier pipeline
│   ├── generate.py             # Local GPU code generation (Qwen2.5-Coder)
│   ├── sandbox_harness.py      # Docker-based sandboxed execution
│   ├── signals.py              # Entropy + consistency feature extraction
│   ├── filter_tasks.py         # BigCodeBench task filtering
│   └── label_taxonomy.py       # LLM-assisted taxonomy labeling
├── data/
│   ├── raw_tasks/              # Filtered BigCodeBench subset
│   ├── generations/            # LLM outputs + logprobs
│   ├── labels/                 # Execution PASS/FAIL + taxonomy labels
│   └── pilot/                  # Hand-written pilot examples
├── tests/
│   └── test_parse_calls.py     # Unit tests for the parser
├── configs/
│   └── default.yaml            # Centralized project configuration
├── docs/                       # Project documentation
├── notebooks/                  # Exploratory analysis
└── paper/                      # LaTeX paper drafts
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the parser (with def-use chain tracking)
python src/parse_calls.py

# Run baselines against 10 Intent Misuse pilot examples
python src/baselines.py

# Run the classifier pipeline (synthetic data)
python src/classifier.py

# Run unit tests
python -m pytest tests/ -v
```

## Current Status

| Component | Status | Notes |
|---|---|---|
| `parse_calls.py` | ✅ Working | v2: now catches assign-then-misuse-later patterns |
| `baselines.py` | ✅ Working | Tests 10 diverse Intent Misuse examples |
| `classifier.py` | ✅ Working | GroupKFold pipeline on synthetic data |
| `generate.py` | ⚠️ Needs GPU | Local GPU generation script |
| `sandbox_harness.py` | ⚠️ Needs Docker | Docker-based execution harness |

## Key Documentation

- [Project Handoff Summary](docs/project_handoff_summary.md) — full project arc and design decisions
- [Literature Survey](docs/literature_survey_hallucination_detection.md) — 21-paper comparison
- [Tech Stack](docs/tech_stack_hallucination_detection.md) — detailed stack spec
- [Execution Roadmap](docs/execution_roadmap.md) — 14-week phased plan
