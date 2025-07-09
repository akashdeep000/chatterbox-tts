import os
from typing import List
from .config import settings

class VoiceManager:
    """
    Manages available cloned voices.
    """
    def __init__(self, voices_dir: str = None):
        self.voices_dir = voices_dir or settings.VOICES_DIR
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir)

    def list_voices(self) -> List[str]:
        """
        Scans the voices directory for available voice files.
        """
        if not os.path.exists(self.voices_dir):
            return []
        return [f for f in os.listdir(self.voices_dir) if os.path.isfile(os.path.join(self.voices_dir, f))]

    def voice_exists(self, voice_id: str) -> bool:
        """
        Checks if a voice with the given ID exists by scanning the directory.
        """
        return voice_id in self.list_voices()

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

    def save_voice(self, voice_id: str, file_contents: bytes):
        """
        Saves a voice file to the voices directory.
        """
        if self.voice_exists(voice_id):
            raise FileExistsError(f"Voice '{voice_id}' already exists.")
        voice_path = os.path.join(self.voices_dir, voice_id)
        with open(voice_path, "wb") as f:
            f.write(file_contents)

    def delete_voice(self, voice_id: str):
        """
        Deletes a voice file from the voices directory.
        """
        voice_path = self.get_voice_path(voice_id)
        os.remove(voice_path)