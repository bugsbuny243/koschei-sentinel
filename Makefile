.PHONY: install check test lint format run dataset-dry-run dataset-release-dry-run benchmark compare

install:
	python -m pip install -e '.[dev]'

check: lint test benchmark compare

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
