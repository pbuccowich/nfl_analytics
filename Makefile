.PHONY: setup requirements install format lint test build clean help

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip pip-tools
	.venv/bin/pip install -e .[dev]

requirements: ## Compile dependencies into requirements.txt
	pip-compile --output-file=requirements.txt pyproject.toml

install: ## Install package locally in editable mode
	pip install -e .

format: ## Auto-format code
	ruff format .

lint: ## Run linter and type checks
	ruff check .

test: ## Run unit tests
	pytest tests/

build: ## Build wheel and source distributions
	python -m build

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +