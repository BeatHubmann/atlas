.PHONY: help build up down logs clean test lint format setup

help:
	@echo "ATLAS Development Commands:"
	@echo "  make setup                  - Initial setup of the infrastructure"
	@echo "  make build                  - Build Docker images (local platform)"
	@echo "  make build-multiplatform    - Build for ARM64 & AMD64 platforms"
	@echo "  make build-multiplatform-push - Build and push multi-platform images"
	@echo "  make up                     - Start all services"
	@echo "  make down                   - Stop all services"
	@echo "  make logs                   - View logs from all services"
	@echo "  make clean                  - Clean up volumes and containers"
	@echo "  make test                   - Run tests"
	@echo "  make lint                   - Run linting"
	@echo "  make format                 - Format code"

setup:
	@bash scripts/setup.sh

build:
	docker compose build

build-multiplatform:
	@bash scripts/build-multiplatform.sh

build-multiplatform-push:
	@bash scripts/build-multiplatform.sh --push

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/
	uv run python -m mypy src/

format:
	uv run black src/ tests/
	uv run ruff check src/ tests/ --fix

# Development commands
dev-api:
	uv run uvicorn atlas_atc.api.main:app --reload --host 0.0.0.0 --port 8000

dev-dashboard:
	uv run streamlit run src/atlas_atc/frontend/dashboard.py

# Database commands
db-shell:
	docker compose exec postgres psql -U atlas -d atlas_atc

redis-cli:
	docker compose exec redis redis-cli

# Monitoring
prometheus:
	open http://localhost:9090

grafana:
	open http://localhost:3000