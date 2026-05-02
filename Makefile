# Makefile for the publication-readiness branch.
# Targets are intentionally cheap by default. Long experiments are explicit.

PY ?= python
PIP ?= $(PY) -m pip

.PHONY: help install install-dev lock test test-unit test-methodology test-integration smoke lint format clean data repro-mini repro-full tables paper-bundle

help:
	@echo "Available targets:"
	@echo "  install            Install runtime + benchmarks + designs deps."
	@echo "  install-dev        Install dev tools (pytest, ruff, pyright, pip-tools)."
	@echo "  lock               Re-compile requirements.lock.txt from requirements.in."
	@echo "  test               Run the full test suite."
	@echo "  test-unit          Run unit tests only."
	@echo "  test-methodology   Run methodology-pinning tests only."
	@echo "  test-integration   Run integration / smoke tests only."
	@echo "  smoke              Run the cheap end-to-end smoke pipeline on synthetic data."
	@echo "  lint               Lint with ruff."
	@echo "  format             Auto-format with ruff."
	@echo "  data               Download MAGIC dataset to data/source/ (verifies SHA-256)."
	@echo "  repro-mini         Reduced reproduction (~ minutes) on synthetic data."
	@echo "  repro-full         Full 30-replica reproduction on MAGIC. Long. Use with care."
	@echo "  tables             Regenerate dissertation-style tables from experiments/."
	@echo "  paper-bundle       Bundle artifacts under experiments/_paper_bundle/."

install:
	$(PIP) install -e ".[benchmarks,designs]"

install-dev:
	$(PIP) install -e ".[benchmarks,designs,dev,notebooks]"

lock:
	$(PY) -m piptools compile --resolver=backtracking -o requirements.lock.txt requirements.in

test:
	$(PY) -m pytest -q

test-unit:
	$(PY) -m pytest tests/unit -q

test-methodology:
	$(PY) -m pytest tests/methodology -q

test-integration:
	$(PY) -m pytest tests/integration -q

smoke:
	$(PY) -m pytest tests/integration -q -m smoke

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

data data-v1:
	$(PY) -m doe_xgb.cli datasets fetch --all

data-checksums:
	$(PY) -m doe_xgb.cli datasets verify-checksums

repro-mini:
	$(PY) -m doe_xgb.cli run --config configs/reduced_repro.yaml

repro-full:
	$(PY) -m doe_xgb.cli run --config configs/dissertation_baseline_xgb_magic.yaml

tables:
	$(PY) scripts/run_aggregate_results.py

paper-bundle:
	@mkdir -p experiments/_paper_bundle
	@echo "Bundle target is a placeholder. Customize once the article structure stabilizes."

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
