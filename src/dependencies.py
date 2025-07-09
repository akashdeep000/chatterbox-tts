from functools import lru_cache
from .tts import TextToSpeechEngine

@lru_cache(maxsize=1)
def get_tts_engine() -> TextToSpeechEngine:
    """
    Returns a cached instance of the TextToSpeechEngine.
    The @lru_cache(maxsize=1) decorator ensures that the engine is initialized only once.
    """
    return TextToSpeechEngine()