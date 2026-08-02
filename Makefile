.PHONY: install check test lint format run

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
