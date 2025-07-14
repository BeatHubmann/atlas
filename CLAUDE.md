# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATLAS (Air Traffic Learning & Analytics System) is a machine learning application for aircraft trajectory prediction. It implements multiple prediction models from simple linear extrapolation to advanced deep learning approaches (LSTM, Transformer).

## Development Commands

### Setup and Dependencies
```bash
# Install base dependencies
uv sync

# Install with ML dependencies (PyTorch, transformers)
uv sync --extra ml

# Install all extras (ml, frontend, dev)
uv sync --all-extras
```

### Running the Application
```bash
# Run API server
atlas-server

# Run Streamlit dashboard
atlas-dashboard

# Train models
atlas-train

# Run evaluation benchmarks
atlas-evaluate

# Using Docker Compose (recommended)
docker-compose up -d
```

### Testing and Code Quality
```bash
# Run tests with coverage
pytest

# Run specific test
pytest tests/test_api.py -v

# Code formatting
black src/ tests/

# Linting
ruff check src/ tests/ --fix

# Type checking
mypy src/
```

## Architecture

### Model Implementation Pattern
All trajectory prediction models inherit from base classes in `src/atlas_atc/models/base.py`:
- `TrajectoryPredictor`: Abstract base for all models
- `TorchTrajectoryPredictor`: Base for PyTorch models with device management
- Models return standardized `PredictionResult` dataclass

### Configuration Management
- Central configuration in `src/atlas_atc/config.py` using Pydantic BaseSettings
- Environment variables override defaults
- Access via: `from atlas_atc.config import settings`

### Database Access
- PostgreSQL with asyncpg for async operations
- Redis for caching
- Connection strings in environment variables

### API Structure
- FastAPI-based REST API in `src/atlas_atc/api/`
- Async request handling
- Standardized response formats

## Key Directories
- `src/atlas_atc/models/`: ML model implementations (Kalman, Linear, LSTM, Transformer)
- `src/atlas_atc/data/`: Data loading and processing utilities
- `src/atlas_atc/evaluation/`: Model benchmarking and metrics
- `data/scat/`: SCAT dataset storage
- `models/checkpoints/`: Trained model storage
- `results/experiments/`: Experiment tracking

## Development Notes
- Python 3.12+ required
- Uses UV package manager (not pip)
- Docker containers for API (port 8000) and dashboard (port 8501)
- PostgreSQL on port 5432, Redis on port 6379
- Pre-commit hooks configured - run `pre-commit install`