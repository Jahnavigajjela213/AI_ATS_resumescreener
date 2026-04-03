import os
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI Resume Screening System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Paths
    MODEL_DIR: str = str(BASE_DIR / "models" / "saved")
    DATA_DIR: str = str(BASE_DIR / "data")
    LOG_DIR: str = str(BASE_DIR / "logs")

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_URL: str = "http://localhost:8000"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(BASE_DIR / "logs" / "app.log")

    # ML
    MAX_FEATURES: int = 5000
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
