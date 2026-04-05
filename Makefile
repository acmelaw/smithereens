.DEFAULT_GOAL := help

# ── Development ───────────────────────────────────────────────

.PHONY: install
install: ## Install all dependencies (dev + test)
	pdm install -G dev -G test

.PHONY: lint
lint: ## Run ruff linter
	pdm run ruff check src/ tests/

.PHONY: format
format: ## Run ruff formatter
	pdm run ruff format src/ tests/

.PHONY: format-check
format-check: ## Check formatting without changes
	pdm run ruff format --check src/ tests/

.PHONY: typecheck
typecheck: ## Run type checking (if mypy is installed)
	pdm run python -m mypy src/ --ignore-missing-imports || true

.PHONY: test
test: ## Run tests
	pdm run pytest

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	pdm run pytest -v

.PHONY: test-cov
test-cov: ## Run tests with coverage
	pdm run pytest --cov=smithereens --cov-report=term-missing

.PHONY: check
check: lint format-check test ## Run all checks (lint + format + test)

# ── Container ─────────────────────────────────────────────────

.PHONY: build
build: ## Build the container image
	docker build -f Containerfile -t smithereens:latest .

.PHONY: run
run: ## Run interactively in a container
	docker compose run --rm smithereens

# ── Helpers ───────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ .pytest_cache/ htmlcov/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.py[codz]' -delete 2>/dev/null || true

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
