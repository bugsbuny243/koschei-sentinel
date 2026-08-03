.PHONY: install install-training check test lint format run dataset-dry-run dataset-release-dry-run benchmark compare together-plan training-plan training-execute

install:
	python -m pip install -e '.[dev]'

install-training:
	python -m pip install -e '.[dev,training]'

check: lint test benchmark compare together-plan training-plan

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

run:
	python -m koschei_sentinel

dataset-dry-run:
	SENTINEL_DATASET_SALT=local-development-salt-only \
		sentinel-dataset-export \
		--input fixtures/arvis.source.safe.json \
		--manifest build/sentinel.dry-run.json \
		--dry-run

dataset-release-dry-run:
	sentinel-dataset-split \
		--input fixtures/dataset.safe.jsonl \
		--dry-run

benchmark:
	sentinel-eval \
		--suite fixtures/evals/suite.safe.jsonl \
		--candidate sentinel-baseline-v0.4 \
		--output build/evals/baseline.json

compare:
	rm -rf build/comparisons/safe
	sentinel-compare \
		--suite fixtures/evals/suite.safe.jsonl \
		--registry fixtures/models/candidates.safe.json \
		--output-dir build/comparisons/safe \
		--require-all

together-plan:
	sentinel-compare \
		--suite fixtures/evals/suite.safe.jsonl \
		--registry fixtures/models/candidates.together.low-cost.json \
		--plan-only

training-plan:
	sentinel-train \
		--config fixtures/training/config.safe.json \
		--plan-output build/training/fixture.plan.json

training-execute:
	sentinel-train \
		--config configs/training/qlora.colab.example.json \
		--plan-output build/training/colab.plan.json \
		--execute
