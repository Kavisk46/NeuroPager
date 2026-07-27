.PHONY: install dev-install lint format typecheck test check clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	black .
	ruff check --fix .

typecheck:
	mypy src

test:
	pytest --cov=neuropager --cov-report=term-missing

check: lint typecheck test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist *.egg-info
