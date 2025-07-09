import io
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
import tempfile
import os
from voice_manager import VoiceManager

class TextToSpeechEngine:
    """
    A wrapper around the Chatterbox TTS library to provide text-to-speech functionality.
    """
    def __init__(self):
        """
        Initializes the TTS engine and loads the pre-trained model.
        """
        print("Initializing TTS Engine...")
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.tts = ChatterboxTTS.from_pretrained(device=device)
        self.voice_manager = VoiceManager()
        print("TTS Engine Initialized.")

    def generate(self, text: str, voice_id: str = None) -> bytes:
        """
        Generates a complete audio file in WAV format.
        """
        print(f"Generating audio for text: '{text}'")

        audio_prompt_path = None
        if voice_id:
            try:
                audio_prompt_path = self.voice_manager.get_voice_path(voice_id)
                print(f"Using voice: {voice_id}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                raise e

        # Generate speech and return it as a byte string.
        audio_tensor = self.tts.generate(text, audio_prompt_path=audio_prompt_path)
        # Use a temporary file to save the audio, then read it back.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            torchaudio.save(tmpfile.name, audio_tensor, self.tts.sr)
            tmpfile.seek(0)
            audio_bytes = tmpfile.read()

        # Clean up the temporary file
        os.remove(tmpfile.name)

        return audio_bytes

    def stream(self, text: str, voice_id: str = None):
        """
        Streams audio in chunks as bytes.
        """
        print(f"Streaming audio for text: '{text}'")
        # Generate the full audio and then stream it in chunks.
        audio_bytes = self.generate(text, voice_id=voice_id)
        chunk_size = 1024
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i+chunk_size]