"""
Manages application-wide dependencies, including the singleton TTS engine instance.
"""
from typing import Optional
from .tts import TextToSpeechEngine
from .voice_manager import VoiceManager

# Global singleton instances
tts_engine: Optional[TextToSpeechEngine] = None
voice_manager: Optional[VoiceManager] = None

def get_tts_engine() -> TextToSpeechEngine:
    """
    Dependency injector that returns the global TTS engine instance.
    Raises an exception if the engine is not initialized.
    """
    if tts_engine is None:
        raise RuntimeError("TTS Engine has not been initialized.")
    return tts_engine

def get_voice_manager() -> VoiceManager:
    """
    Dependency injector that returns the global VoiceManager instance.
    Raises an exception if the manager is not initialized.
    """
    if voice_manager is None:
        raise RuntimeError("VoiceManager has not been initialized.")
    return voice_manager