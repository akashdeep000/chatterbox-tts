"""
Main Text-to-Speech (TTS) engine module for production.

This module defines the primary `TextToSpeechEngine` class, which orchestrates
the TTS process. It integrates model loading, voice conditioning, caching,
and audio generation with performance optimizations.
"""

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Generator, AsyncGenerator
import io
import gc
import asyncio
import time
import functools

# Third-party imports
import librosa
import torch
import torch.nn.functional as F
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

from .config import settings, tts_config
from .voice_manager import VoiceManager
from .text_processing import split_text_into_chunks, punc_norm
from .logging_config import log


@dataclass
class Conditionals:
    """Holds conditioning information for TTS models."""
    t3: T3Cond
    gen: dict

    def to(self, device: str):
        """Moves tensors to the specified device."""
        self.t3 = self.t3.to(device=torch.device(device))
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=torch.device(device))
        return self


@dataclass
class SynthesisParams:
    """Holds parameters for the synthesis process."""
    text_tokens: torch.Tensor
    conds: Conditionals
    cfg_weight: float
    temperature: float
    tokens_per_slice: int
    remove_milliseconds: int
    remove_milliseconds_start: int
    chunk_overlap_method: Literal["zero", "full"]
    loop: asyncio.AbstractEventLoop
    text_chunk_count: int
    next_chunk_event: asyncio.Event


class _AudioProcessor:
    """Handles audio processing tasks like WAV header creation and PCM conversion."""

    @staticmethod
    def create_wav_header(sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
        """Creates a WAV file header for streaming."""
        data_size = 2**31 - 1 - 44  # Max size for a 32-bit signed integer
        header = io.BytesIO()
        header.write(b'RIFF')
        header.write((data_size + 36).to_bytes(4, 'little'))
        header.write(b'WAVEfmt ')
        header.write((16).to_bytes(4, 'little'))
        header.write((1).to_bytes(2, 'little'))
        header.write(channels.to_bytes(2, 'little'))
        header.write(sample_rate.to_bytes(4, 'little'))
        header.write((sample_rate * channels * sample_width).to_bytes(4, 'little'))
        header.write((channels * sample_width).to_bytes(2, 'little'))
        header.write((sample_width * 8).to_bytes(2, 'little'))
        header.write(b'data')
        header.write(data_size.to_bytes(4, 'little'))
        header.seek(0)
        return header.read()

    @staticmethod
    def to_pcm(audio_tensor: torch.Tensor) -> bytes:
        """Converts an audio tensor to PCM byte data."""
        audio_tensor_clamped = torch.clamp(audio_tensor.cpu(), -1.0, 1.0)
        audio_tensor_int = (audio_tensor_clamped * 32767).to(torch.int16)
        pcm_data = audio_tensor_int.numpy().tobytes()
        safe_delete_tensors(audio_tensor, audio_tensor_clamped, audio_tensor_int)
        return pcm_data


class TextToSpeechEngine:
    """
    The main engine for Text-to-Speech synthesis.
    This class manages the entire TTS pipeline, including model loading,
    voice conditioning, audio generation, and streaming.
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self):
        """
        Initializes the TTS engine, loads models, and sets the computation device.
        It automatically detects and uses CUDA or MPS if available, otherwise falls back to CPU.
        """
        # Auto-detect best available device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.fp16 = tts_config.enable_fp16 and self.device == "cuda"
        self.dtype = torch.float16 if self.fp16 else torch.float32

        # Load the core ChatterboxTTS model from a local path
        self.tts = OriginalChatterboxTTS.from_local(
            settings.MODEL_PATH,
            device=self.device
        )
        if self.fp16:
            self.tts = self.tts.half()
        # Initialize voice manager for handling different voices
        self.voice_manager = VoiceManager()
        # Store model conditionals, initially from the loaded model
        self.voice_cache: dict[str, Conditionals] = {}
        # Cache for the path of the audio prompt to avoid reprocessing
        self._cached_audio_prompt_path: Optional[str] = None
        # Set the sample rate from the model configuration
        self.sr = self.tts.sr
        self.audio_processor = _AudioProcessor()

    def clear_voice_cache(self, voice_id: Optional[str] = None):
        """Clears the voice cache."""
        if voice_id and voice_id in self.voice_cache:
            del self.voice_cache[voice_id]
        elif not voice_id:
            self.voice_cache.clear()

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """Prepares conditioning information from a reference audio file."""
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

        if self.fp16:
            ve_embed = ve_embed.half()

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=torch.device(self.device))

        voice_id = Path(wav_fpath).name
        self.voice_cache[voice_id] = Conditionals(t3_cond, s3gen_ref_dict)

    async def _t3_producer_task(
        self,
        text_chunks: list,
        speech_token_queue: asyncio.Queue,
        params: SynthesisParams,
    ):
        """Producer task for T3 model. Generates speech tokens and puts them into a queue."""
        num_chunks = len(text_chunks)
        loop = params.loop
        try:
            for i, chunk in enumerate(text_chunks):
                if i > 0:
                    # Wait for the signal from the consumer before starting the next chunk.
                    await params.next_chunk_event.wait()
                    params.next_chunk_event.clear()

                log.info(f"T3: Processing text chunk {i+1}/{num_chunks}")
                # 1. Text to Tokens
                text_tokens = await loop.run_in_executor(
                    None, self.tts.tokenizer.text_to_tokens, punc_norm(chunk)
                )
                if params.cfg_weight > 0.0:
                    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
                sot, eot = self.tts.t3.hp.start_text_token, self.tts.t3.hp.stop_text_token
                text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)

                # 2. T3 Inference (blocking)
                log.info(f"T3: Starting inference for chunk {i+1}/{num_chunks}")
                t3_inference_gen = self.tts.t3.inference(
                    t3_cond=params.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=500,
                    temperature=params.temperature,
                    cfg_weight=params.cfg_weight,
                )

                # 3. Buffer all tokens from the generator for the current chunk
                token_stream = []
                for speech_token_batch in t3_inference_gen:
                    token_stream.extend(speech_token_batch.squeeze(0))

                # 4. Slice the buffered tokens and queue them with metadata
                slices = []
                while len(token_stream) >= params.tokens_per_slice:
                    slices.append(token_stream[:params.tokens_per_slice])
                    token_stream = token_stream[params.tokens_per_slice:]
                if token_stream:
                    slices.append(token_stream)

                total_slices = len(slices)
                # Define the trigger point for the consumer to signal back.
                # It's the lower of 50% or the second-to-last slice, but at least 1.
                trigger_point = min(total_slices // 2, max(1, total_slices - 1))

                log.info(f"T3: Finished inference for chunk {i+1}/{num_chunks}, created {total_slices} slices. Trigger point is slice {trigger_point}.")

                for slice_idx, token_slice in enumerate(slices):
                    await speech_token_queue.put(
                        (torch.stack(token_slice), i + 1, slice_idx + 1, total_slices, trigger_point)
                    )
        except Exception as e:
            log.error(f"Error in T3 producer task: {e}", exc_info=True)
        finally:
            await speech_token_queue.put(None) # Signal end of production

    async def _s3gen_consumer_task(
        self,
        speech_token_queue: asyncio.Queue,
        audio_chunk_queue: asyncio.Queue,
        params: SynthesisParams,
    ):
        """Consumer task for S3Gen model. Converts speech tokens to audio chunks."""
        loop = params.loop
        previous_length = 0
        try:
            while True:
                queue_item = await speech_token_queue.get()
                if queue_item is None:
                    break

                token_chunk, text_chunk_num, slice_num, total_slices, trigger_point = queue_item
                log.info(
                    f"S3Gen: Starting inference for slice {slice_num}/{total_slices} "
                    f"(from text chunk {text_chunk_num}/{params.text_chunk_count})"
                )

                # 1. Prepare tokens for S3Gen
                speech_tokens = token_chunk.unsqueeze(0)
                eos_token = torch.tensor([self.tts.t3.hp.stop_text_token]).unsqueeze(0).to(self.device)
                tokens_with_eos = torch.cat([speech_tokens, eos_token], dim=1)
                speech_tokens = tokens_with_eos[0]
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)
                if speech_tokens.shape[-1] < 3:
                    padding_needed = 3 - speech_tokens.shape[-1]
                    speech_tokens = F.pad(speech_tokens, (0, padding_needed), "constant", 0)

                # 2. S3Gen Inference (blocking)
                partial_inference = functools.partial(
                    self.tts.s3gen.inference,
                    speech_tokens=speech_tokens,
                    ref_dict=params.conds.gen
                )
                wav, _ = await loop.run_in_executor(None, partial_inference)
                wav_gpu = wav.squeeze(0).detach()

                # 3. Post-processing
                if params.chunk_overlap_method == "full":
                    wav_gpu = wav_gpu[previous_length:]
                if params.remove_milliseconds > 0:
                    trim_samples = int(self.sr * params.remove_milliseconds / 1000)
                    wav_gpu = wav_gpu[:-trim_samples]
                if params.remove_milliseconds_start > 0:
                    trim_samples_start = int(self.sr * params.remove_milliseconds_start / 1000)
                    wav_gpu = wav_gpu[trim_samples_start:]

                previous_length += wav_gpu.shape[0]

                # 4. Queue audio chunk
                pcm_data = self.audio_processor.to_pcm(wav_gpu)
                await audio_chunk_queue.put(pcm_data)
                log.info(
                    f"S3Gen: Finished inference for slice {slice_num}/{total_slices} "
                    f"(from text chunk {text_chunk_num}/{params.text_chunk_count})"
                )
                speech_token_queue.task_done()

                # If the trigger point is reached, signal the producer to start the next T3 inference.
                if slice_num == trigger_point and text_chunk_num < params.text_chunk_count:
                    log.info(f"S3Gen: Reached trigger point for chunk {text_chunk_num}. Signaling producer for next chunk.")
                    params.next_chunk_event.set()

        except Exception as e:
            log.error(f"Error in S3Gen consumer task: {e}", exc_info=True)
        finally:
            await audio_chunk_queue.put(None) # Signal end of consumption

    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        exaggeration: float = tts_config.exaggeration,
        cfg_weight: float = tts_config.cfg_weight,
        temperature: float = tts_config.temperature,
        text_chunk_size: Optional[int] = tts_config.text_chunk_size,
        tokens_per_slice: Optional[int] = tts_config.tokens_per_slice,
        remove_milliseconds: int = tts_config.remove_milliseconds,
        remove_milliseconds_start: int = tts_config.remove_milliseconds_start,
        chunk_overlap_method: Literal["zero", "full"] = tts_config.chunk_overlap_method,
        start_time: Optional[float] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Streams synthesized audio using a producer-consumer pattern."""
        loop = asyncio.get_running_loop()
        audio_prompt_path = self.voice_manager.get_voice_path(voice_id) if voice_id else None
        conds = await self._prepare_and_get_conds(audio_prompt_path, exaggeration, loop)

        yield self.audio_processor.create_wav_header(self.sr)

        text_chunks = split_text_into_chunks(text, text_chunk_size)
        if not text_chunks:
            return

        log.info(f"Text chunked into {len(text_chunks)} parts. Starting TTS stream.")

        # This queue is the buffer between the T3 (producer) and S3Gen (consumer) stages.
        # Its size is critical for the proactive signaling to be effective. It holds the remaining
        # slices of the current text chunk, allowing the S3Gen consumer to stay busy while the
        # T3 producer is already running the expensive inference for the *next* text chunk.
        # A larger size allows for larger text chunks without stalling the pipeline.
        speech_token_queue = asyncio.Queue(maxsize=10)

        # This queue buffers the final audio chunks before they are yielded to the client.
        # It helps to smooth out the delivery by handling any minor timing variations.
        audio_chunk_queue = asyncio.Queue(maxsize=2)

        # This event synchronizes the producer and consumer, allowing the producer to start the
        # next chunk's T3 inference proactively when the consumer is partway through the current one.
        next_chunk_event = asyncio.Event()

        params = SynthesisParams(
            text_tokens=None, conds=conds, cfg_weight=cfg_weight, temperature=temperature,
            tokens_per_slice=tokens_per_slice, remove_milliseconds=remove_milliseconds,
            remove_milliseconds_start=remove_milliseconds_start,
            chunk_overlap_method=chunk_overlap_method, loop=loop, text_chunk_count=len(text_chunks),
            next_chunk_event=next_chunk_event
        )

        # Start producer and consumer tasks
        producer_task = loop.create_task(
            self._t3_producer_task(text_chunks, speech_token_queue, params)
        )
        consumer_task = loop.create_task(
            self._s3gen_consumer_task(speech_token_queue, audio_chunk_queue, params)
        )

        # Stream audio chunks to the client
        first_chunk_time = None
        while True:
            audio_chunk = await audio_chunk_queue.get()
            if first_chunk_time is None and audio_chunk is not None:
                first_chunk_time = time.time()
                if start_time:
                    log.info(f"Time to first audio chunk: {first_chunk_time - start_time:.4f}s")

            if audio_chunk is None:
                break
            yield audio_chunk
            audio_chunk_queue.task_done()

        await asyncio.gather(producer_task, consumer_task)
        log.info("Finished TTS stream.")

    async def _prepare_and_get_conds(self, audio_prompt_path: Optional[str], exaggeration: float, loop: asyncio.AbstractEventLoop) -> Conditionals:
        """Prepares and retrieves the appropriate conditionals."""
        voice_id = Path(audio_prompt_path).name if audio_prompt_path else "default"
        if audio_prompt_path and voice_id not in self.voice_cache:
            log.info(f"Voice '{voice_id}' not in cache. Preparing new conditionals...")
            await loop.run_in_executor(None, self.prepare_conditionals, audio_prompt_path, exaggeration)
            log.info(f"Finished preparing conditionals for '{voice_id}'.")
        elif audio_prompt_path:
            log.info(f"Using cached conditionals for voice '{voice_id}'.")

        conds = self.voice_cache.get(voice_id)
        if conds is None:
            if self.tts.conds:
                conds = self.tts.conds
            else:
                raise ValueError("No audio prompt provided, and no default conditionals are loaded.")

        current_exaggeration_tensor = exaggeration * torch.ones(1, 1, 1, device=self.device, dtype=self.dtype)
        if not torch.equal(conds.t3.emotion_adv, current_exaggeration_tensor):
            _cond: T3Cond = conds.t3
            conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=current_exaggeration_tensor,
            ).to(device=self.device)
        return conds
