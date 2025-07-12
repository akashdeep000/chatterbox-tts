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
import asyncio
import time
import functools
import logging

 # Third-party imports
import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import numpy as np

# Local application/library specific imports
from .utils import safe_delete_tensors
from chatterbox.models.t3 import T3
from chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens
from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts import ChatterboxTTS as OriginalChatterboxTTS

from .config import settings
from .voice_manager import VoiceManager
from .text_processing import split_text_into_chunks, punc_norm


# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class Conditionals:
    """
    A data class to hold the conditioning information for the TTS models.
    This class encapsulates the tensors required for conditioning the T3 and S3Gen models.
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        """
        Moves all tensor attributes to the specified device.

        Args:
            device: The target device (e.g., 'cuda', 'cpu').

        Returns:
            The Conditionals object with tensors moved to the specified device.
        """
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        """
        Saves the conditionals to a file using torch.save.

        Args:
            fpath: The file path where the conditionals will be saved.
        """
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        """
        Loads conditionals from a file.

        Args:
            fpath: The file path from which to load the conditionals.
            map_location: The device to map the loaded tensors to.

        Returns:
            A new Conditionals object with the loaded data.
        """
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class TextToSpeechEngine:
    """
    The main engine for Text-to-Speech synthesis.

    This class manages the entire TTS pipeline, including model loading,
    voice conditioning, audio generation, and streaming.
    """
    # Constants for conditioning audio length
    ENC_COND_LEN = 6 * S3_SR      # 6 seconds for encoder conditioning
    DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds for decoder conditioning

    def __init__(self):
        """
        Initializes the TTS engine, loads models, and sets the computation device.
        It automatically detects and uses CUDA or MPS if available, otherwise falls back to CPU.
        """
        logger.info("Initializing TTS Engine...")
        # Auto-detect best available device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"TTS Engine using device: {self.device}")

        # Load the core ChatterboxTTS model from a local path
        self.tts = OriginalChatterboxTTS.from_local(
            settings.MODEL_PATH,
            device=self.device
        )
        # Initialize voice manager for handling different voices
        self.voice_manager = VoiceManager()
        # Store model conditionals, initially from the loaded model
        self.voice_cache: dict[str, Conditionals] = {}
        # Cache for the path of the audio prompt to avoid reprocessing
        self._cached_audio_prompt_path: Optional[str] = None
        # Set the sample rate from the model configuration
        self.sr = self.tts.sr
        logger.info("TTS Engine Initialized.")

    def clear_voice_cache(self, voice_id: Optional[str] = None):
        """
        Clears the voice cache. If a voice_id is provided, only that voice is removed.
        Otherwise, the entire cache is cleared.
        """
        if voice_id:
            if voice_id in self.voice_cache:
                del self.voice_cache[voice_id]
                logger.info(f"Voice '{voice_id}' cleared from cache.")
        else:
            self.voice_cache.clear()
            self._cached_audio_prompt_path = None
            logger.info("Voice cache cleared.")

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """
        Prepares the conditioning information from a reference audio file.

        This involves loading the audio, resampling it for different models,
        and generating embeddings and tokens for conditioning.

        Args:
            wav_fpath: Path to the reference audio file.
            exaggeration: The degree of emotional exaggeration to apply.
        """
        # Load and resample reference audio for S3Gen and T3 models
        s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        # Trim audio to the required conditioning length
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        # Generate reference embeddings for the S3Gen model
        s3gen_ref_dict = self.tts.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Generate speech prompt tokens for the T3 model if required
        t3_cond_prompt_tokens = None
        if plen := self.tts.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.tts.s3gen.tokenizer
            tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(tokens).to(self.device)

        # Generate voice encoder embeddings
        ve_embed = torch.from_numpy(self.tts.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        # Create T3 conditioning object
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

        # Store the combined conditionals in the cache
        voice_id = Path(wav_fpath).name
        self.voice_cache[voice_id] = Conditionals(t3_cond, s3gen_ref_dict)
        self._cached_audio_prompt_path = str(Path(wav_fpath).resolve())
        logger.info(f"Conditionals processed and cached for voice: {voice_id}")

    async def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        tokens_per_slice: int = 25,
        remove_milliseconds: int = 45,
        remove_milliseconds_start: int = 25,
        chunk_overlap_method: Literal["zero", "full"] = "zero",
    ) -> Generator[torch.Tensor, None, None]:
        voice_id = Path(audio_prompt_path).name if audio_prompt_path else "default"
        loop = asyncio.get_running_loop()

        if audio_prompt_path and voice_id not in self.voice_cache:
            logger.info(f"Voice '{voice_id}' not in cache. Processing: {audio_prompt_path}")
            await loop.run_in_executor(None, self.prepare_conditionals, audio_prompt_path, exaggeration)

        conds = self.voice_cache.get(voice_id)

        if conds is None:
            if self.tts.conds:
                conds = self.tts.conds
            else:
                raise ValueError("No audio prompt provided, and no default conditionals are loaded. Please provide `audio_prompt_path`.")

        current_exaggeration_tensor = exaggeration * torch.ones(1, 1, 1, device=self.device)
        if not torch.equal(conds.t3.emotion_adv, current_exaggeration_tensor):
            logger.info(f"Updating emotion exaggeration to: {exaggeration}")
            _cond: T3Cond = conds.t3
            conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=current_exaggeration_tensor,
            ).to(device=self.device)

        text = punc_norm(text)
        text_tokens = await loop.run_in_executor(
            None, self.tts.tokenizer.text_to_tokens, text
        )
        text_tokens = text_tokens.to(self.device)


        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot, eot = self.tts.t3.hp.start_text_token, self.tts.t3.hp.stop_text_token
        text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)

        with torch.inference_mode():
            # Nested function to stream speech tokens from the T3 model.
            async def _t3_infer():
                partial_inference = functools.partial(
                    self.tts.t3.inference,
                    t3_cond=conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=1000,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                )
                tokens_tensor = await loop.run_in_executor(
                    None, partial_inference
                )
                for token in tokens_tensor:
                    yield token

            # Nested function to convert speech tokens to a WAV audio tensor.
            async def speech_to_wav(speech_tokens, previous_length=0):
                speech_tokens = speech_tokens[0]
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)

                if speech_tokens.shape[-1] < 3:
                    padding_needed = 3 - speech_tokens.shape[-1]
                    speech_tokens = torch.nn.functional.pad(speech_tokens, (0, padding_needed), "constant", 0)

                partial_inference = functools.partial(
                    self.tts.s3gen.inference,
                    speech_tokens=speech_tokens,
                    ref_dict=conds.gen
                )
                wav, _ = await loop.run_in_executor(
                    None, partial_inference
                )

                wav_gpu = wav.squeeze(0).detach()

                if chunk_overlap_method == "full":
                    wav_gpu = wav_gpu[previous_length:]

                if remove_milliseconds > 0:
                    trim_samples = int(self.sr * remove_milliseconds / 1000)
                    wav_gpu = wav_gpu[:-trim_samples]
                if remove_milliseconds_start > 0:
                    trim_samples_start = int(self.sr * remove_milliseconds_start / 1000)
                    wav_gpu = wav_gpu[trim_samples_start:]

                return wav_gpu.unsqueeze(0), wav_gpu.shape[0] + previous_length

            eos_token = torch.tensor([self.tts.t3.hp.stop_text_token]).unsqueeze(0).to(self.device)

            # Nested function to chunk the token stream for processing.
            async def chunked():
                token_stream = []
                async for batch in _t3_infer():
                    token_stream.extend(batch.squeeze(0))
                    while len(token_stream) >= tokens_per_slice:
                        yield token_stream[:tokens_per_slice]
                        token_stream = token_stream[tokens_per_slice:]
                if token_stream:
                    yield token_stream

            # Nested function to accumulate token chunks.
            async def accumulating_chunks():
                accumulated = []
                async for batch in _t3_infer():
                    accumulated.extend(batch.squeeze(0))
                    if len(accumulated) % tokens_per_slice == 0 and len(accumulated) > 0:
                        yield accumulated.copy()
                if accumulated and len(accumulated) % tokens_per_slice != 0:
                    yield accumulated.copy()

            previous_length = 0
            iterator = chunked() if chunk_overlap_method == "zero" else accumulating_chunks()
            async for slice_tokens in iterator:
                tokens = torch.stack(slice_tokens).unsqueeze(0)
                tokens_with_eos = torch.cat([tokens, eos_token], dim=1)
                wav, previous_length = await speech_to_wav(tokens_with_eos, previous_length)
                yield wav

    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        text_chunk_size: Optional[int] = 100,
        tokens_per_slice: Optional[int] = 25,
        remove_milliseconds: int = 45,
        remove_milliseconds_start: int = 25,
        start_time: Optional[float] = None,
    ) -> Generator[bytes, None, None]:
        audio_prompt_path = self.voice_manager.get_voice_path(voice_id) if voice_id else None

        header = self._create_wav_header(self.sr)
        yield header

        text_chunks = split_text_into_chunks(text, text_chunk_size)
        logger.info(f"Streaming {len(text_chunks)} text chunks.")

        first_chunk_generated = False

        for i, chunk in enumerate(text_chunks):
            logger.debug(f"Generating audio for chunk {i+1}/{len(text_chunks)}")

            # Generate audio for the current text chunk.
            audio_generator = self.generate(
                text=chunk,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                tokens_per_slice=tokens_per_slice,
                remove_milliseconds=remove_milliseconds,
                remove_milliseconds_start=remove_milliseconds_start,
            )

            async for audio_tensor in audio_generator:
                if not first_chunk_generated and start_time:
                    ttfb = time.time() - start_time
                    logger.info(f"Time to first audio chunk: {ttfb:.4f}s")
                    first_chunk_generated = True

                if audio_tensor is not None and audio_tensor.numel() > 0:
                    if hasattr(audio_tensor, 'cpu'):
                        audio_tensor = audio_tensor.cpu()

                    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
                    audio_tensor_int = (audio_tensor * 32767).to(torch.int16)

                    pcm_data = audio_tensor_int.numpy().tobytes()
                    yield pcm_data

                    safe_delete_tensors(audio_tensor, audio_tensor_int)
                    del pcm_data

            # Perform garbage collection periodically to manage memory.
            if i > 0 and i % 3 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _create_wav_header(self, sample_rate, channels=1, sample_width=2, data_size=2**31-1-44):
        """
        Creates a WAV file header for streaming.

        A large data_size is used to indicate a virtually infinite stream.

        Args:
            sample_rate: The audio sample rate.
            channels: The number of audio channels.
            sample_width: The width of each sample in bytes.
            data_size: The size of the data chunk.

        Returns:
            The WAV header in bytes.
        """
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