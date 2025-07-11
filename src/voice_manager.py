import os
from typing import List
from .config import settings

class VoiceManager:
    """
    Manages available cloned voices.
    """
    def __init__(self, voices_dir: str = None, preloaded_voices_dir: str = None):
        self.voices_dir = voices_dir or settings.VOICES_DIR
        self.preloaded_voices_dir = preloaded_voices_dir or settings.PRELOADED_VOICES_DIR

        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir)
        if not os.path.exists(self.preloaded_voices_dir):
            os.makedirs(self.preloaded_voices_dir)

    def list_voices(self) -> List[str]:
        """
        Scans both the preloaded and user-uploaded voices directories for available voice files.
        """
        preloaded_voices = []
        if os.path.exists(self.preloaded_voices_dir):
            preloaded_voices = [f for f in os.listdir(self.preloaded_voices_dir) if os.path.isfile(os.path.join(self.preloaded_voices_dir, f))]

        user_voices = []
        if os.path.exists(self.voices_dir):
            user_voices = [f for f in os.listdir(self.voices_dir) if os.path.isfile(os.path.join(self.voices_dir, f))]

        # Combine and remove duplicates, giving priority to user-uploaded voices
        return sorted(list(set(user_voices + preloaded_voices)))

    def voice_exists(self, voice_id: str) -> bool:
        """
        Checks if a voice with the given ID exists in either directory.
        """
        return self.get_voice_path(voice_id) is not None

    def get_voice_path(self, voice_id: str) -> str:
        """
        Returns the file path for a given voice ID, checking both directories.
        """
        # Prioritize user-uploaded voices
        user_voice_path = os.path.join(self.voices_dir, voice_id)
        if os.path.exists(user_voice_path):
            return user_voice_path

        preloaded_voice_path = os.path.join(self.preloaded_voices_dir, voice_id)
        if os.path.exists(preloaded_voice_path):
            return preloaded_voice_path

        return None

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
        Deletes a voice file from the user-uploaded voices directory.
        Preloaded voices cannot be deleted.
        """
        voice_path = os.path.join(self.voices_dir, voice_id)
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice '{voice_id}' not found in user directory.")

        os.remove(voice_path)