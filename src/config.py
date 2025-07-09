import os
from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    """
    Application configuration settings, loaded from environment variables.
    """
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Logging configuration
    LOG_LEVEL: str = "INFO"

    # TTS and Voice settings
    VOICES_DIR: str = "voices/"
    MODEL_DEVICE: str = "cpu"  # Default to CPU, can be overridden

    # Security
    API_KEY: str  # No default value, must be set in environment

    # CORS settings
    CORS_ORIGINS: list[str] = ["*"] # Allows all origins by default

    class Config:
        # Load from a .env file if it exists
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Instantiate the config object to be used across the application
settings = AppConfig()