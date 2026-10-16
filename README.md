# AI/ML-Based Intelligent Load Balancer

A Layer-7 HTTP load balancer that uses machine learning to predict per-server response times and route requests to the fastest backend. Built as a semester project with a 14-week timeline.

## Repository Layout

| Directory | Purpose |
|-----------|---------|
| `balancer/` | Reverse proxy, decision engine, routing algorithms |
| `backend/` | FastAPI/Flask backend apps with `/metrics` and `/health` endpoints |
| `agent/` | psutil metrics agent running inside each backend container |
| `ml/` | Feature engineering, model training, inference |
| `bench/` | Benchmark suites, Locust traffic scripts, run matrix |
| `docs/` | Hypotheses, design docs, architecture diagrams |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Start the full stack (requires Docker Compose)
docker compose up
```

## Milestones

| Milestone | Week | Focus |
|-----------|------|-------|
| M1 | 1 | Project skeleton and CI green |
| M2 | 3 | Baseline balancer works end-to-end |
| M3 | 5 | Live metrics and logs flowing |
| M4 | 6 | Dataset v1 (200k+ rows) |
| M5 | 8 | Model beats naive baseline by 30%+ MAE |
| M6 | 9 | MVP: ML balancer demo |
| M7 | 11 | Feature freeze, all tests green |
| M8 | 12 | Raw results complete |
| M9 | 14 | Final submission |

## License
MIT