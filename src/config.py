import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class AppConfig(BaseSettings):
    """
    Application configuration settings, loaded from environment variables.
    """
    # Server settings
    HOST: str = Field(
        default="0.0.0.0",
        description="Host address for the application server."
    )
    PORT: int = Field(
        default=8000,
        description="Port for the application server."
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode for the application server."
    )
    # Logging configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level for the application (e.g., INFO, DEBUG, WARNING, ERROR)."
    )

    # TTS and Voice settings
    VOICES_DIR: str = Field(
        default="voices/",
        description="Directory where custom voices are stored."
    )
    PRELOADED_VOICES_DIR: str = Field(
        default="preloaded-voices/",
        description="Directory for preloaded voices."
    )
    MODEL_PATH: str = Field(
        default="models",
        description="Path to the directory containing TTS models."
    )

    # Security
    API_KEY: str = Field(
        description="API key for authentication. Must be set via environment variable."
    )

    # CORS settings
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="List of allowed origins for CORS. Use '*' to allow all origins."
    )

    CONCURRENT_REQUESTS_PER_GPU: int = Field(
        default=1,
        description="Maximum number of concurrent TTS requests to process per GPU."
    )

    CPU_WORKER_COUNT: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="Number of worker processes for the CPU-bound task executor. Defaults to the number of available CPU cores."
    )

    class Config:
        # Load from a .env file if it exists
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'



class TTSConfig(BaseSettings):
    """
    Default TTS parameter values, configurable via environment variables.
    """
    model_config = SettingsConfigDict(env_prefix='TTS_', case_sensitive=False)

    # Voice modulation parameters
    VOICE_EXAGGERATION_FACTOR: float = Field(
        default=0.5,
        description="Controls the voice_exaggeration_factor of the voice. Higher values increase expressiveness."
    )
    CFG_GUIDANCE_WEIGHT: float = Field(
        default=0.5,
        description="Classifier-free guidance weight. Influences how strongly the model adheres to the text prompt."
    )
    SYNTHESIS_TEMPERATURE: float = Field(
        default=0.8,
        description="synthesis_temperature for text-to-speech synthesis. Higher values lead to more varied but potentially less coherent output."
    )

    # Text processing and chunking
    TEXT_PROCESSING_CHUNK_SIZE: int = Field(
        default=150,
        description="Maximum number of characters per text chunk for processing."
    )
    AUDIO_TOKENS_PER_SLICE: int = Field(
        default=35,
        description="Number of audio tokens per slice during streaming synthesis."
    )

    # Silence/pause handling
    REMOVE_LEADING_MILLISECONDS: int = Field(
        default=0,
        description="Duration in milliseconds to remove from the start of generated audio."
    )
    REMOVE_TRAILING_MILLISECONDS: int = Field(
        default=0,
        description="Duration in milliseconds to remove from the end of generated audio."
    )

    # Audio chunk overlap and crossfade
    CHUNK_OVERLAP_STRATEGY: str = Field(
        default="full",
        description="Strategy for overlapping audio chunks: 'full' (overlap and crossfade) or 'zero' (no overlap)."
    )
    CROSSFADE_DURATION_MILLISECONDS: int = Field(
        default=30,
        description="Duration in milliseconds for crossfading between audio chunks"
    )

    # Queue sizes for streaming
    SPEECH_TOKEN_QUEUE_MAX_SIZE: int = Field(
        default=2,
        description="Maximum size of the speech token queue used in streaming. Smaller values reduce initial latency."
    )
    PCM_CHUNK_QUEUE_MAX_SIZE: int = Field(
        default=3,
        description="Maximum size of the PCM chunk queue used in streaming. Smaller values reduce initial latency but may increase risk of stuttering."
    )

# Instantiate the config objects to be used across the application
settings = AppConfig()
tts_config = TTSConfig()