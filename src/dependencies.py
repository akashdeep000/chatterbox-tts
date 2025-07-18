"""
Manages application-wide dependencies, including the singleton TTS engine manager.
"""
from typing import Optional
from .tts_streaming import TTSEngineManager
from .voice_manager import VoiceManager

# Global singleton instances
tts_engine_manager: Optional[TTSEngineManager] = None
voice_manager: Optional[VoiceManager] = None

def get_tts_engine_manager() -> TTSEngineManager:
    """
    Dependency injector that returns the global TTS engine manager instance.
    Raises an exception if the manager is not initialized.
    """
    if tts_engine_manager is None:
        raise RuntimeError("TTS Engine Manager has not been initialized.")
    return tts_engine_manager

def get_voice_manager() -> VoiceManager:
    """
    Dependency injector that returns the global VoiceManager instance.
    Raises an exception if the manager is not initialized.
    """
    if voice_manager is None:
        raise RuntimeError("VoiceManager has not been initialized.")
    return voice_manager