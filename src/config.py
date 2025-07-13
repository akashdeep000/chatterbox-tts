import os
from pydantic_settings import BaseSettings
from pydantic import BaseModel

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
    PRELOADED_VOICES_DIR: str = "preloaded-voices/"
    MODEL_PATH: str = "models"

    # Security
    API_KEY: str  # No default value, must be set in environment

    # CORS settings
    CORS_ORIGINS: list[str] = ["*"] # Allows all origins by default

    class Config:
        # Load from a .env file if it exists
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

class TTSConfig(BaseModel):
    """
    Default TTS parameter values.
    """
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    text_chunk_size: int = 75
    tokens_per_slice: int = 25
    remove_milliseconds: int = 15
    remove_milliseconds_start: int = 10
    chunk_overlap_method: str = "zero"
    enable_fp16: bool = True # Use half-precision for faster inference on compatible GPUs

# Instantiate the config objects to be used across the application
settings = AppConfig()
tts_config = TTSConfig()