import os
from typing import List

class VoiceManager:
    """
    Manages available cloned voices.
    """
    def __init__(self, voices_dir: str = "voices/"):
        self.voices_dir = voices_dir
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir)
        self._available_voices = self._scan_voices()
        print(f"Available voices: {self._available_voices}")

    def _scan_voices(self) -> List[str]:
        """
        Scans the voices directory for available voice files.
        """
        if not os.path.exists(self.voices_dir):
            return []
        return [f for f in os.listdir(self.voices_dir) if os.path.isfile(os.path.join(self.voices_dir, f))]

    def voice_exists(self, voice_id: str) -> bool:
        """
        Checks if a voice with the given ID exists.
        """
        return voice_id in self._available_voices

    def get_voice_path(self, voice_id: str) -> str:
        """
        Returns the file path for a given voice ID.
        """
        if not self.voice_exists(voice_id):
            raise FileNotFoundError(f"Voice '{voice_id}' not found.")

        voice_path = os.path.join(self.voices_dir, voice_id)
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice file not found at path: {voice_path}")

        return voice_path