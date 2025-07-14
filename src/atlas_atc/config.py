"""Configuration management for ATLAS."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    
    # Database configuration
    DATABASE_URL: str = Field(
        default="postgresql://atlas:atlas_password@localhost:5432/atlas_atc",
        env="DATABASE_URL"
    )
    
    # Redis configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL"
    )
    
    # API configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    
    # ML configuration
    DEVICE: str = Field(default="mps", env="DEVICE")  # mps, cuda, cpu
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    MAX_SEQUENCE_LENGTH: int = Field(default=100, env="MAX_SEQUENCE_LENGTH")
    
    # Prediction horizons (minutes)
    PREDICTION_HORIZONS: list[int] = [2, 5, 8]
    
    # Monitoring
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    GRAFANA_PORT: int = Field(default=3000, env="GRAFANA_PORT")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

