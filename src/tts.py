"""
Main Text-to-Speech (TTS) engine module.

This module defines the primary `TextToSpeechEngine` class, which orchestrates
the TTS process. It integrates model loading, voice conditioning, caching,
and audio generation with performance optimizations.
"""

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Generator
import io

import gc
from .utils import safe_delete_tensors
# Third-party imports
import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import numpy as np

# Local application/library specific imports
from chatterbox.models.t3 import T3
from chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens
from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts import ChatterboxTTS as OriginalChatterboxTTS

from .config import settings
from .voice_manager import VoiceManager
from .text_processing import split_text_for_streaming, punc_norm


@dataclass
class Conditionals:
    """
    A data class to hold the conditioning information for the TTS models.
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        """Moves all tensor attributes to the specified device."""
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        """Saves the conditionals to a file."""
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        """Loads conditionals from a file."""
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class TextToSpeechEngine:
    """
    The main engine for Text-to-Speech synthesis.
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self):
        """Initializes the TTS engine, loads models, and sets the device."""
        print("Initializing TTS Engine...")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"TTS Engine using device: {self.device}")

        self.tts = OriginalChatterboxTTS.from_local(
            settings.MODEL_PATH,
            device=self.device
        )
        self.voice_manager = VoiceManager()
        self.conds: Optional[Conditionals] = self.tts.conds
        self._cached_audio_prompt_path: Optional[str] = None
        self.sr = self.tts.sr
        print("TTS Engine Initialized.")

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """
        Prepares the conditioning information from a reference audio file.
        """
        s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.tts.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        t3_cond_prompt_tokens = None
        if plen := self.tts.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.tts.s3gen.tokenizer
            tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(tokens).to(self.device)

        ve_embed = torch.from_numpy(self.tts.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        self._cached_audio_prompt_path = str(Path(wav_fpath).resolve())
        print(f"INFO: Conditionals processed and cached for audio: {self._cached_audio_prompt_path}")

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> Optional[torch.Tensor]:
        """
        Generates an audio tensor for a given text chunk.
        """
        if audio_prompt_path:
            normalized_provided_path = str(Path(audio_prompt_path).resolve())
            if normalized_provided_path != self._cached_audio_prompt_path:
                print(f"INFO: New or different audio prompt. Processing: {normalized_provided_path}")
                self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            else:
                print(f"INFO: Audio prompt '{normalized_provided_path}' matches cache. Reusing conditionals.")

        if self.conds is None:
            raise ValueError("No audio prompt provided, and no default conditionals are loaded. Please provide `audio_prompt_path`.")

        current_exaggeration_tensor = exaggeration * torch.ones(1, 1, 1, device=self.device)
        if not torch.equal(self.conds.t3.emotion_adv, current_exaggeration_tensor):
            print(f"INFO: Updating emotion exaggeration to: {exaggeration}")
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=current_exaggeration_tensor,
            ).to(device=self.device)

        text = punc_norm(text)
        if not text:
            return None

        text_tokens = self.tts.tokenizer.text_to_tokens(text).to(self.device)
        if len(text_tokens) == 0:
            return None

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot, eot = self.tts.t3.hp.start_text_token, self.tts.t3.hp.stop_text_token
        text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)


        speech_tokens = self.tts.t3.inference(
            t3_cond=self.conds.t3, text_tokens=text_tokens,
            max_new_tokens=1000, temperature=temperature, cfg_weight=cfg_weight,
        )

        clean_tokens = drop_invalid_tokens(speech_tokens[0]).to(self.device)
        if len(clean_tokens) == 0:
            return None

        wav, _ = self.tts.s3gen.inference(
            speech_tokens=clean_tokens, ref_dict=self.conds.gen
        )
        return wav.squeeze(0).cpu().detach()

    def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> Generator[bytes, None, None]:
        """
        Generates streaming audio from text by splitting it into chunks.
        """
        audio_prompt_path = self.voice_manager.get_voice_path(voice_id) if voice_id else None

        header = self._create_wav_header(self.sr)
        yield header

        text_chunks = split_text_for_streaming(text)
        print(f"Streaming {len(text_chunks)} text chunks.")

        for i, chunk in enumerate(text_chunks):
            print(f"Generating audio for chunk {i+1}/{len(text_chunks)}")
            with torch.no_grad():
                audio_tensor = self.generate(
                    text=chunk,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )

                if audio_tensor is not None and audio_tensor.numel() > 0:
                    # Ensure tensor is on CPU for streaming
                    if hasattr(audio_tensor, 'cpu'):
                        audio_tensor = audio_tensor.cpu()

                    # Clamp values to [-1, 1] before conversion
                    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
                    audio_tensor_int = (audio_tensor * 32767).to(torch.int16)

                    # Yield the raw audio data as bytes
                    pcm_data = audio_tensor_int.numpy().tobytes()
                    yield pcm_data

                    # Clean up this chunk
                    safe_delete_tensors(audio_tensor, audio_tensor_int)
                    del pcm_data

            # Periodic memory cleanup during generation
            if i > 0 and i % 3 == 0:  # Every 3 chunks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _create_wav_header(self, sample_rate, channels=1, sample_width=2, data_size=2**31-1-44):
            header = io.BytesIO()
            header.write(b'RIFF' + (data_size + 36).to_bytes(4, 'little') + b'WAVEfmt ' +
                         (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') +
                         channels.to_bytes(2, 'little') + sample_rate.to_bytes(4, 'little') +
                         (sample_rate * channels * sample_width).to_bytes(4, 'little') +
                         (channels * sample_width).to_bytes(2, 'little') +
                         (sample_width * 8).to_bytes(2, 'little') + b'data' +
                         data_size.to_bytes(4, 'little'))
            header.seek(0)
            return header.read()