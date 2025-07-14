# ATLAS: Air Traffic Learning & Analytics System

## Executive Summary

ATLAS is a trajectory prediction system designed for air traffic control applications. The system implements multiple prediction algorithms ranging from classical Kalman filtering to state-of-the-art deep learning architectures, providing comparative analysis capabilities for operational deployment scenarios.

## System Architecture

### Core Components

- **Prediction Engine**: Multi-model trajectory prediction framework supporting linear extrapolation, Kalman filtering, LSTM networks, and Transformer architectures
- **Data Processing Pipeline**: High-throughput ingestion and preprocessing of SCAT (Swedish Civil Air Traffic) datasets with ASTERIX CAT062 format support
- **API Service**: RESTful interface for real-time prediction requests and system monitoring
- **Visualization Dashboard**: Interactive Streamlit-based interface for flight trajectory analysis with comprehensive legends and flight plan comparisons
- **Evaluation Framework**: Comprehensive benchmarking suite for model performance assessment
- **Monitoring Infrastructure**: Prometheus/Grafana stack for operational metrics and system health monitoring

### Technical Specifications

- **Runtime Environment**: Python 3.12+
- **Container Platform**: Docker with multi-architecture support (ARM64/AMD64)
  - Docker buildx for cross-platform builds
  - Supports both Apple Silicon and x86_64 architectures
- **Database Systems**: PostgreSQL 16 (primary storage), Redis 7 (caching layer)
- **ML Frameworks**: PyTorch 2.7+, Transformers 4.35+
- **API Framework**: FastAPI with async/await support
- **Package Management**: UV for deterministic dependency resolution
- **Code Quality Tools**: Black (formatting), Ruff (linting), MyPy (type checking)

## Deployment Procedures

### Prerequisites

- Docker Engine 20.10+ with Compose V2
- 16GB RAM minimum (32GB recommended for ML workloads)
- 50GB available storage for datasets and model checkpoints

### Initial Setup

1. Clone repository and navigate to project root:
```bash
git clone https://github.com/BeatHubmann/atlas.git
cd atlas
```

2. Execute automated setup:
```bash
make setup
```

This process will:
- Create required directory structure
- Initialize environment configuration
- Build container images
- Deploy service stack
- Verify system health

### Docker Build Options

```bash
make build                      # Build for local platform only
make build-multiplatform        # Build for ARM64 & AMD64 platforms
make build-multiplatform-push   # Build and push multi-platform images to registry
```

The multi-platform build uses Docker buildx to create images compatible with both Apple Silicon (ARM64) and x86_64 (AMD64) architectures.

### Service Endpoints

Upon successful deployment, the following services are accessible:

- API Service: `http://localhost:8000`
- Dashboard Interface: `http://localhost:8501` - Interactive flight trajectory visualization with:
  - Real-time flight path rendering with legends
  - Altitude and performance profiles
  - Flight plan vs actual trajectory comparison
  - Waypoint visualization
  - Debug information for data quality analysis
- PostgreSQL Database: `localhost:5432`
- Redis Cache: `localhost:6379`
- Prometheus Metrics: `http://localhost:9090`
- Grafana Dashboards: `http://localhost:3000` (default login: admin/admin)

### Container Resource Limits

- **API Service**: 8GB memory limit, 4GB reserved
- **All services**: Configured with health checks and auto-restart policies

## Operational Commands

### Service Management
```bash
make up        # Deploy service stack
make down      # Terminate services
make logs      # Monitor service logs
make clean     # Remove all containers and volumes
```

### Development Operations
```bash
make test      # Execute test suite with pytest
make lint      # Run ruff and mypy checks
make format    # Apply black and ruff formatting
```

### Development Mode
```bash
make dev-api       # Run API server in development mode with hot reload
make dev-dashboard # Run Streamlit dashboard in development mode
```

### Database Access
```bash
make db-shell   # Access PostgreSQL shell
make redis-cli  # Access Redis CLI
```

### Monitoring Access
```bash
make prometheus # Open Prometheus UI (http://localhost:9090)
make grafana    # Open Grafana dashboards (http://localhost:3000)
```

## Model Implementations

### Available Algorithms

1. **Linear Extrapolation**: Baseline predictor using constant velocity assumption
2. **Kalman Filter**: Classical state estimation with configurable process/measurement noise
3. **LSTM Network**: Sequence-to-sequence architecture with attention mechanism
4. **Transformer**: Self-attention based model optimized for trajectory sequences

### Training Procedures

Execute model training via dedicated CLI:
```bash
atlas-train --model lstm --epochs 100 --batch-size 32
```

### Evaluation Metrics

- Position Error (meters): Euclidean distance between predicted and actual positions
- Along-Track Error: Error component parallel to flight path
- Cross-Track Error: Error component perpendicular to flight path
- Prediction Horizon: Maximum lookahead time with acceptable error bounds

## Data Specifications

### SCAT Dataset Format

The system processes SCAT (Swedish Civil Air Traffic) data with the following schema:
- Aircraft ID and callsign
- Timestamp (UTC) with millisecond precision
- Position (latitude, longitude, altitude) from ASTERIX I062/105
- Velocity vectors (vx, vy) from ASTERIX I062/185
- Flight level and vertical rate from ASTERIX I062/136 and I062/220
- Aircraft derived data (heading, IAS, Mach) from ASTERIX I062/380
- Flight plan information (departure, destination, route)
- Predicted trajectory waypoints with estimated times

### Data Pipeline

1. Raw data ingestion from JSON/CSV/Parquet formats
2. ASTERIX CAT062 format parsing for surveillance data
3. Coordinate transformation to local tangent plane
4. Trajectory segmentation and filtering  
5. Feature engineering for ML models
6. Train/validation/test splitting

## Performance Benchmarks

Baseline performance metrics on standard SCAT dataset:
- Linear Model: 150m average position error at 60s horizon
- Kalman Filter: 95m average position error at 60s horizon
- LSTM: 72m average position error at 60s horizon
- Transformer: 68m average position error at 60s horizon

## Security Considerations

- All services run as non-privileged users within containers
- Database credentials isolated via environment variables
- API authentication ready (implementation pending)
- Network segmentation between service tiers

## Maintenance Procedures

### Database Backup
```bash
docker compose exec postgres pg_dump -U atlas atlas_atc > backup.sql
```

### Log Rotation
Logs automatically rotate based on size/age policies defined in Docker daemon configuration.

### Monitoring Alerts
Prometheus alerting rules available in `monitoring/alerts/` directory.

## Code Quality Standards

The project enforces strict code quality standards:

- **Formatting**: Black with 88-character line length
- **Linting**: Ruff with comprehensive rule set including:
  - pycodestyle (E, W)
  - pyflakes (F)
  - isort (I)
  - flake8-bugbear (B)
  - flake8-comprehensions (C4, C408)
  - pyupgrade (UP)
- **Type Checking**: MyPy with strict mode enabled
- **Testing**: Pytest with coverage reporting

All code must pass quality checks:
```bash
make lint    # Run ruff and mypy
make format  # Auto-format with black and ruff
make test    # Run test suite with coverage
```

## Contributing

Contributions must adhere to project coding standards. Execute pre-commit hooks:
```bash
pre-commit install
```

## License

MIT License - See LICENSE file for details.

## Point of Contact

For technical inquiries regarding system architecture or operational issues, submit via project issue tracker.