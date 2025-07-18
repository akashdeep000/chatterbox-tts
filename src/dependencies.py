"""
Manages application-wide dependencies, including the singleton TTS engine manager.
"""
from typing import Optional
import pysbd
from .tts_streaming import TTSEngineManager
from .voice_manager import VoiceManager

# Global singleton instances
tts_engine_manager: Optional[TTSEngineManager] = None
voice_manager: Optional[VoiceManager] = None
segmenter: Optional[pysbd.Segmenter] = None

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

def get_segmenter() -> pysbd.Segmenter:
    """
    Dependency injector that returns the global pysbd.Segmenter instance.
    Raises an exception if the segmenter is not initialized.
    """
    if segmenter is None:
        raise RuntimeError("Segmenter has not been initialized.")
    return segmenter