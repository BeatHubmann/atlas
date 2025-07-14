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

## Data source
The data source is described at https://www.sciencedirect.com/science/article/pii/S2352340923003591

## Data Descriptions

### SCAT Dataset Overview
- **Swedish Civil Air Traffic Control (SCAT) dataset**
- 13 weeks of continuous data (Oct 2016 - Sep 2017)
- ~170,000 flights from Swedish Flight Information Region
- Data from Malmö (ESMM) and Stockholm (ESOS) control centers
- Only scheduled flights (no military/private/incidents)

### Dataset Structure

#### 1. Flight Files (e.g., `100000.json`)
Each JSON file contains data for a single flight with:

**Top-level fields:**
- `id`: Unique flight identifier
- `centre_ctrl`: Control center information
- `fpl`: Flight plan data
- `plots`: Surveillance/radar data
- `predicted_trajectory`: Future trajectory predictions

**Flight Plan (`fpl`) structure:**
- `fpl_base`: Basic flight info
  - `callsign`: Aircraft call sign
  - `aircraft_type`: Aircraft model (e.g., B738, A320)
  - `adep`: Departure airport (ICAO code)
  - `ades`: Destination airport (ICAO code)
  - `wtc`: Wake turbulence category
  - `flight_rules`: Flight rules (I=IFR, V=VFR)
  - `equip_status_rvsm`: RVSM equipment status
- `fpl_arr`: Arrival information
  - `approach_clearance`: Boolean
  - `arrival_runway`: Assigned runway
  - `star`: Standard Terminal Arrival Route
- `fpl_dep`: Departure information
- `fpl_clearance`: ATC clearances
  - `cfl`: Cleared flight level
  - `assigned_heading_val`: Assigned heading
  - `assigned_speed_val`: Assigned speed
- `fpl_plan_update`: Route updates
- `fpl_holding`: Holding pattern info

**Surveillance Data (`plots`) - ASTERIX CAT062 format:**
- `time_of_track`: Timestamp for each position
- `I062/105`: Position data
  - `lat`: Latitude (decimal degrees, WGS-84)
  - `lon`: Longitude (decimal degrees, WGS-84)
- `I062/136`: Altitude
  - `measured_flight_level`: Altitude in flight levels (100s of feet)
- `I062/185`: Velocity
  - `vx`: East-west velocity (m/s, positive=east)
  - `vy`: North-south velocity (m/s, positive=north)
- `I062/200`: Mode of movement
  - `adf`: Altitude discrepancy flag
  - `long`: Longitudinal acceleration (0=constant, 1=increasing, 2=decreasing)
  - `trans`: Transversal acceleration (0=straight, 1=right turn, 2=left turn)
  - `vert`: Vertical movement (0=level, 1=climb, 2=descent)
- `I062/220`: Vertical rate
  - `rocd`: Rate of climb/descent (feet/minute, negative=descent)
- `I062/380`: Aircraft derived data
  - `subitem3`: Magnetic heading (degrees)
  - `subitem6`: Selected altitude
  - `subitem7`: Final state selected altitude
  - `subitem13`: Barometric vertical rate
  - `subitem26`: Indicated airspeed (IAS)
  - `subitem27`: Mach number

**Predicted Trajectory (`predicted_trajectory`):**
Array of predicted route segments with:
- `route`: Array of waypoints containing:
  - `fix_name`: Waypoint identifier
  - `fix_kind`: Type (AIRPORT, RP, GP, ENT_POINT, etc.)
  - `lat`/`lon`: Predicted position
  - `eto`: Estimated time over
  - `afl_value`: Actual flight level
  - `rfl_value`: Requested flight level
  - `is_ato`: Whether actively tracked

#### 2. Airspace File (`airspace.json`)
- Named waypoints with coordinates
- Control sector boundaries
- Valid for entire week period

#### 3. Weather File (`grib_meteo.json`)
- Wind and temperature forecasts
- Grid-based data at multiple flight levels
- From World Meteorological Organization (WMO)
- Updated every 3 hours

### Data Processing Notes
- Times in UTC, ISO 8601 format
- Coordinates in WGS-84
- Altitudes in flight levels (100 feet units)
- Speeds in various units (m/s for ground speed, knots for IAS)
- Headings in degrees (magnetic)

### Key Features for ML/Analysis
- Complete flight trajectories with 4D positions (lat, lon, alt, time)
- Aircraft performance data (speeds, climb rates, headings)
- Flight plan vs. actual trajectory comparison
- ATC clearances and route modifications
- Weather context available
- Multi-sensor fused surveillance data (ARTAS)
