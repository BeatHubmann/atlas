"""Configuration management for ATLAS."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"

    # Database configuration
    DATABASE_URL: str = Field(
        default="postgresql://atlas:atlas_password@localhost:5432/atlas_atc"
    )

    # Redis configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379"
    )

    # API configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)

    # ML configuration
    DEVICE: str = Field(default="mps")  # mps, cuda, cpu
    BATCH_SIZE: int = Field(default=32)
    MAX_SEQUENCE_LENGTH: int = Field(default=100)

    # Prediction horizons (minutes)
    PREDICTION_HORIZONS: list[int] = [2, 5, 8]

    # Monitoring
    PROMETHEUS_PORT: int = Field(default=9090)
    GRAFANA_PORT: int = Field(default=3000)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "allow"
    }


# Global settings instance
settings = Settings()

