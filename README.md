# ATLAS: Air Traffic Learning & Analytics System

## Executive Summary

ATLAS is a trajectory prediction system designed for air traffic control applications. The system implements multiple prediction algorithms ranging from classical Kalman filtering to state-of-the-art deep learning architectures, providing comparative analysis capabilities for operational deployment scenarios.

## System Architecture

### Core Components

- **Prediction Engine**: Multi-model trajectory prediction framework supporting linear extrapolation, Kalman filtering, LSTM networks, and Transformer architectures
- **Data Processing Pipeline**: High-throughput ingestion and preprocessing of SCAT (Simulated Continuous Air Traffic) datasets
- **API Service**: RESTful interface for real-time prediction requests and system monitoring
- **Evaluation Framework**: Comprehensive benchmarking suite for model performance assessment
- **Monitoring Infrastructure**: Prometheus/Grafana stack for operational metrics and system health monitoring

### Technical Specifications

- **Runtime Environment**: Python 3.12+
- **Container Platform**: Docker with multi-architecture support (ARM64/AMD64)
- **Database Systems**: PostgreSQL 16 (primary storage), Redis 7 (caching layer)
- **ML Frameworks**: PyTorch 2.7+, Transformers 4.35+
- **API Framework**: FastAPI with async/await support
- **Package Management**: UV for deterministic dependency resolution

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

### Service Endpoints

Upon successful deployment, the following services are accessible:

- API Service: `http://localhost:8000`
- Dashboard Interface: `http://localhost:8501`
- PostgreSQL Database: `localhost:5432`
- Redis Cache: `localhost:6379`
- Prometheus Metrics: `http://localhost:9090`
- Grafana Dashboards: `http://localhost:3000`

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
make test      # Execute test suite
make lint      # Run code quality checks
make format    # Apply code formatting standards
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

The system processes SCAT (Simulated Continuous Air Traffic) data with the following schema:
- Aircraft ID
- Timestamp (UTC)
- Position (latitude, longitude, altitude)
- Velocity vectors
- Flight phase indicators

### Data Pipeline

1. Raw data ingestion from CSV/Parquet formats
2. Coordinate transformation to local tangent plane
3. Trajectory segmentation and filtering
4. Feature engineering for ML models
5. Train/validation/test splitting

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

## Contributing

Contributions must adhere to project coding standards. Execute pre-commit hooks:
```bash
pre-commit install
```

## License

MIT License - See LICENSE file for details.

## Point of Contact

For technical inquiries regarding system architecture or operational issues, submit via project issue tracker.