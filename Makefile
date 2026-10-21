# One-command entry points for the milestones.
#
# Windows without make: run the command on the right-hand side directly.
# The campaign itself is resumable; `make benchmark` continues an interrupted
# matrix instead of starting over.

PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: help up down logs test benchmark analyse train ablations stack-check

help:
	@echo "make up          build and start the stack (dashboard on :9090)"
	@echo "make test        run the automated suite"
	@echo "make benchmark   run/resume the 125-cell Phase 7 matrix"
	@echo "make analyse     tables, CIs and charts from the matrix results"
	@echo "make train       retrain on data/ and write models/"
	@echo "make ablations   feature ablations + unseen-profile tests"
	@echo "make down        stop the stack"

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f balancer

test:
	$(PYTHON) -m pytest -q

benchmark:
	$(PYTHON) bench/run_matrix.py --repeats 5 --duration 60 --warmup 12

analyse:
	$(PYTHON) bench/analyse_matrix.py

train:
	$(PYTHON) ml/train.py --data data --output models

ablations:
	$(PYTHON) ml/train.py --output scratch/ablations --only xgboost --no-cv --no-artifacts --tag _full
	$(PYTHON) ml/train.py --output scratch/ablations --only xgboost --no-cv --no-artifacts --tag _no_trend --drop-features cpu_delta_5s,rt_p95_5s_ms
	$(PYTHON) ml/train.py --output scratch/ablations --only xgboost --no-cv --no-artifacts --tag _no_req_type --drop-features req_type_light,req_type_cpu,req_type_io,req_type_payload
	$(PYTHON) ml/generalisation.py --profile burst
	$(PYTHON) ml/generalisation.py --profile wave

stack-check:
	$(PYTHON) bench/check_stack.py
