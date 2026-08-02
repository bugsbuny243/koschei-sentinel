.PHONY: install check test lint format run dataset-dry-run dataset-release-dry-run

install:
	python -m pip install -e '.[dev]'

check: lint test

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
