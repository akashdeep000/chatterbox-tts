"""
Main Text-to-Speech (TTS) engine module for production with enhanced streaming.

This module defines the primary `TextToSpeechEngine` class, which orchestrates
the TTS process. It integrates model loading, voice conditioning, caching,
and audio generation with performance optimizations.
"""

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Literal, Optional, Generator, AsyncGenerator, AsyncIterator, List
import io
import gc
import asyncio
import time
import functools
from enum import Enum
import random

from .audio_encoding import AudioEncoder


class InitializationState(Enum):
    """Represents the initialization state of the TTS engine."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


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
from .text_processing import split_text_into_chunks
from .logging_config import log


async def async_generator_wrapper(sync_generator: Generator):
    """
    Wraps a synchronous generator to make it asynchronously iterable.
    Each item is yielded using asyncio.to_thread to prevent blocking the event loop.
    """
    _END_OF_GENERATOR = object() # Local sentinel to signal generator exhaustion

    def _get_next_item_safe():
        """Synchronous helper to get the next item or signal exhaustion."""
        try:
            return next(sync_generator)
        except StopIteration:
            return _END_OF_GENERATOR

    loop = asyncio.get_running_loop()
    while True:
        # Run the synchronous helper in a separate thread to avoid blocking the event loop
        item = await loop.run_in_executor(None, _get_next_item_safe)
        if item is _END_OF_GENERATOR:
            break
        yield item


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
    cfg_guidance_weight: float
    synthesis_temperature: float
    text_processing_chunk_size: int
    audio_tokens_per_slice: int
    remove_trailing_milliseconds: int
    remove_leading_milliseconds: int
    chunk_overlap_strategy: Literal["zero", "full"]
    crossfade_duration_milliseconds: int
    loop: asyncio.AbstractEventLoop
    text_chunk_count: int
    request_id: str


class _AudioProcessor:
    """Handles audio processing tasks like WAV header creation and PCM conversion."""

    @staticmethod
    async def to_pcm(audio_tensor: torch.Tensor, loop: asyncio.AbstractEventLoop) -> bytes:
        """
        Converts an audio tensor to PCM byte data asynchronously to avoid blocking.
        The tensor is moved to the CPU and converted inside a thread pool executor.
        """
        def _blocking_conversion():
            """The synchronous part of the conversion."""
            audio_tensor_clamped = torch.clamp(audio_tensor.cpu(), -1.0, 1.0)
            audio_tensor_int = (audio_tensor_clamped * 32767).to(torch.int16)
            pcm_data = audio_tensor_int.numpy().tobytes()
            safe_delete_tensors(audio_tensor, audio_tensor_clamped, audio_tensor_int)
            return pcm_data

        # Offload the blocking CPU operations to a separate thread
        process_pool = loop.get_default_executor()
        if hasattr(loop, 'process_pool'):
            process_pool = loop.process_pool
        return await loop.run_in_executor(process_pool, _blocking_conversion)


class TextToSpeechEngine:
    """
    The main engine for Text-to-Speech synthesis.
    This class manages the entire TTS pipeline, including model loading,
    voice conditioning, audio generation, and streaming.
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, gpu_id: int = 0):
        """
        Initializes the TTS engine, setting up basic attributes.
        Model loading and device setup are handled asynchronously by `ainit`.
        """
        self.gpu_id = gpu_id
        self.device = None # Will be set during async initialization
        self.tts = None # Will be loaded during async initialization
        self.voice_manager = VoiceManager()
        self.voice_cache: dict[str, Conditionals] = {}
        self._cached_audio_prompt_path: Optional[str] = None
        self.sr = S3GEN_SR # Default, will be updated after model loads
        self.audio_processor = _AudioProcessor()
        self._initialization_state: InitializationState = InitializationState.NOT_STARTED
        self._initialization_progress: str = ""
        self._initialization_error: Optional[str] = None
        self.tts_semaphore = asyncio.Semaphore(settings.CONCURRENT_REQUESTS_PER_GPU)
        self._cache_lock = asyncio.Lock()

    async def ainit(self):
        """
        Asynchronously initializes the TTS engine, loads models, and sets the computation device.
        It automatically detects and uses CUDA or MPS if available, otherwise falls back to CPU.
        """
        try:
            self._initialization_state = InitializationState.INITIALIZING
            self._initialization_progress = "Validating configuration..."

            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = f"cuda:{self.gpu_id}"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


            log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
            log.info(f"{log_prefix} Initializing Chatterbox TTS model...")
            log.info(f"{log_prefix} Device: {self.device}, Model path: {settings.MODEL_PATH}")
            if self.device == "cpu":
                log.warning(f"{log_prefix} WARNING: No CUDA or MPS device found. Falling back to CPU. Performance will be significantly degraded.")


            self._initialization_progress = "Configuring device compatibility..."
            # Patch torch.load for CPU compatibility if needed
            if self.device == 'cpu':
                original_load = torch.load
                original_load_file = None

                # Try to patch safetensors if available
                try:
                    import safetensors.torch
                    original_load_file = safetensors.torch.load_file
                except ImportError:
                    pass

                def force_cpu_torch_load(f, map_location=None, **kwargs):
                    # Always force CPU mapping if we're on a CPU device
                    return original_load(f, map_location='cpu', **kwargs)

                def force_cpu_load_file(filename, device=None):
                    # Force CPU for safetensors loading too
                    return original_load_file(filename, device='cpu')

                torch.load = force_cpu_torch_load
                if original_load_file:
                    safetensors.torch.load_file = force_cpu_load_file

            self._initialization_progress = "Loading TTS model (this may take a while)..."
            # Initialize model with run_in_executor for non-blocking
            loop = asyncio.get_event_loop()
            self.tts = await loop.run_in_executor(
                None,
                lambda: OriginalChatterboxTTS.from_local(
                    settings.MODEL_PATH,
                    device=self.device
                )
            )
            self.sr = self.tts.sr # Set sample rate from the loaded model

            self._initialization_state = InitializationState.READY
            self._initialization_progress = "Model ready"
            self._initialization_error = None
            log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
            log.info(f"✓ {log_prefix} Model initialized successfully on {self.device}")
            return self.tts

        except Exception as e:
            self._initialization_state = InitializationState.ERROR
            self._initialization_error = str(e)
            self._initialization_progress = f"Failed: {str(e)}"
            log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
            log.error(f"✗ {log_prefix} Failed to initialize model: {e}")
            raise e

    def get_initialization_status(self) -> dict:
        """Returns the current initialization status of the TTS engine."""
        return {
            "state": self._initialization_state.value,
            "progress": self._initialization_progress,
            "error": self._initialization_error
        }

    def clear_voice_cache(self, voice_id: Optional[str] = None):
        """Clears the voice cache."""
        if voice_id and voice_id in self.voice_cache:
            del self.voice_cache[voice_id]
        elif not voice_id:
            self.voice_cache.clear()

    def prepare_conditionals(self, wav_fpath: str, voice_exaggeration_factor: float = 0.5):
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


        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=voice_exaggeration_factor * torch.ones(1, 1, 1),
        ).to(device=torch.device(self.device))

        voice_id = Path(wav_fpath).name
        self.voice_cache[voice_id] = Conditionals(t3_cond, s3gen_ref_dict)

    async def _prepare_and_get_conds(self, audio_prompt_path: Optional[str], voice_exaggeration_factor: float, loop: asyncio.AbstractEventLoop, request_id: str) -> Conditionals:
        """Prepares and retrieves the appropriate conditionals."""
        log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
        voice_id = Path(audio_prompt_path).name if audio_prompt_path else "default"

        # Use a lock to prevent race conditions when preparing the same voice multiple times
        async with self._cache_lock:
            if audio_prompt_path and voice_id not in self.voice_cache:
                log.info(f"{log_prefix}[{request_id}] Voice '{voice_id}' not in cache. Preparing new conditionals...")
                await loop.run_in_executor(None, self.prepare_conditionals, audio_prompt_path, voice_exaggeration_factor)
                log.info(f"{log_prefix}[{request_id}] Finished preparing conditionals for '{voice_id}'.")
            elif audio_prompt_path:
                log.info(f"{log_prefix}[{request_id}] Using cached conditionals for voice '{voice_id}'.")

        conds = self.voice_cache.get(voice_id)
        if conds is None:
            if self.tts.conds:
                conds = self.tts.conds
            else:
                raise ValueError("No audio prompt provided, and no default conditionals are loaded.")

        current_exaggeration_tensor = voice_exaggeration_factor * torch.ones(1, 1, 1, device=self.device)
        if not torch.equal(conds.t3.emotion_adv, current_exaggeration_tensor):
            _cond: T3Cond = conds.t3
            conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=current_exaggeration_tensor,
            ).to(device=self.device)
        return conds

    def _blocking_t3_inference(
        self,
        t3_cond: "T3Cond",
        text_tokens: torch.Tensor,
        synthesis_temperature: float,
        cfg_guidance_weight: float,
        stream: torch.cuda.Stream,
    ) -> Generator[torch.Tensor, None, None]:
        """Runs the blocking T3 inference and returns a generator for the token stream."""
        with torch.cuda.stream(stream):
            # Directly return the generator from tts.t3.inference_stream
            return self.tts.t3.inference_stream(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                max_new_tokens=1000, # Consider making this configurable if needed
                temperature=synthesis_temperature,
                cfg_weight=cfg_guidance_weight,
            )

    async def _t3_producer_task(
        self,
        text_chunks: list,
        speech_token_queue: asyncio.Queue,
        params: SynthesisParams,
        t3_stream: torch.cuda.Stream,
        s3gen_stream: torch.cuda.Stream,
    ):
        """Producer task for T3 model. Generates speech tokens and puts them into a queue."""
        num_chunks = len(text_chunks)
        loop = params.loop
        try:
            for i, chunk in enumerate(text_chunks):
                is_first_text_chunk = (i == 0)
                is_last_text_chunk = (i == num_chunks - 1)

                log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                log.info(f"{log_prefix}[{params.request_id}] T3: Processing text chunk {i+1}/{num_chunks}")
                # 1. Text to Tokens
                process_pool = loop.get_default_executor()
                if hasattr(loop, 'process_pool'):
                    process_pool = loop.process_pool
                text_tokens = await loop.run_in_executor(
                    process_pool, self.tts.tokenizer.text_to_tokens, chunk
                )
                # Move text_tokens to the correct device
                text_tokens = text_tokens.to(self.device)

                if params.cfg_guidance_weight > 0.0:
                    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
                sot, eot = self.tts.t3.hp.start_text_token, self.tts.t3.hp.stop_text_token
                text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)

                # 2. T3 Inference (blocking)
                log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                log.info(f"{log_prefix}[{params.request_id}] T3: Starting inference for chunk {i+1}/{num_chunks}")
                t3_start_time = time.time()
                # 3. T3 Inference (non-blocking)
                sync_token_generator = await loop.run_in_executor(
                    None,
                    self._blocking_t3_inference,
                    params.conds.t3,
                    text_tokens,
                    params.synthesis_temperature,
                    params.cfg_guidance_weight,
                    t3_stream,
                )
                t3_inference_time = time.time() - t3_start_time
                log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                log.info(f"{log_prefix}[{params.request_id}] T3: Inference for chunk {i+1}/{num_chunks} took {t3_inference_time:.4f}s")

                # 4. Stream individual tokens and accumulate into slices before queuing
                current_slice = [] # array of single token [token shape: (B, 1)]
                slice_idx = 0

                # Define the look-ahead buffer size (20% of audio_tokens_per_slice or 10 tokens, whichever is larger)
                look_ahead_buffer_size = max(3, int(0.2 * params.audio_tokens_per_slice))
                # Create an iterator from the async generator
                token_iterator = async_generator_wrapper(sync_token_generator).__aiter__()
                generator_exhausted = False # Initialize the flag

                while True:
                    try:
                        # Fill up current_slice as tokens come
                        current_slice.append(await token_iterator.__anext__())
                    except StopAsyncIteration:
                        # Generator is exhausted. Process remaining tokens.
                        generator_exhausted = True

                    # If current_slice length is larger than (look_ahead_buffer_size + audio_tokens_per_slice)
                    # send first audio_tokens_per_slice tokens to s3gen queue.
                    if len(current_slice) >= (look_ahead_buffer_size + params.audio_tokens_per_slice):
                        slice_to_send = current_slice[:params.audio_tokens_per_slice]
                        current_slice = current_slice[params.audio_tokens_per_slice:]
                        # Concatenate all predicted tokens along the sequence dimension.
                        predicted_tokens = torch.cat(slice_to_send, dim=1)  # shape: (B, num_tokens)
                        slice_idx += 1
                        is_first_slice = (slice_idx == 1)
                        # Record an event on the T3 stream after the slice is produced
                        event = torch.cuda.Event()
                        event.record(t3_stream)

                        await speech_token_queue.put(
                            (predicted_tokens, i + 1, slice_idx, is_first_slice, False, is_first_text_chunk, is_last_text_chunk, event)
                        )
                    elif generator_exhausted and current_slice:
                        # If no more tokens are coming, send full current_slice to queue
                        slice_idx += 1
                        is_first_slice = (slice_idx == 1)
                        # Concatenate all predicted tokens along the sequence dimension.
                        predicted_tokens = torch.cat(current_slice, dim=1)  # shape: (B, num_tokens)
                        # Record an event on the T3 stream for the final slice
                        event = torch.cuda.Event()
                        event.record(t3_stream)

                        await speech_token_queue.put(
                            (predicted_tokens, i + 1, slice_idx, is_first_slice, True, is_first_text_chunk, is_last_text_chunk, event) # shape: (B, num_tokens)
                        )
                        current_slice = [] # Clear current_slice after sending
                        break # All tokens sent for this chunk

                    # Break condition: if generator is exhausted and all buffered tokens have been sent
                    if generator_exhausted and not current_slice:
                        break

                log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                log.info(f"{log_prefix}[{params.request_id}] T3: Finished inference for chunk {i+1}/{num_chunks}, produced {slice_idx} slices.")

        except Exception as e:
            log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
            log.error(f"{log_prefix}[{params.request_id}] Error in T3 producer task: {e}", exc_info=True)
        finally:
            await speech_token_queue.put(None) # Signal end of production

    async def _s3gen_consumer_task(
        self,
        speech_token_queue: asyncio.Queue,
        pcm_chunk_queue: asyncio.Queue,
        params: SynthesisParams,
        s3gen_stream: torch.cuda.Stream,
    ):
        """Consumer task for S3Gen model. Converts speech tokens to audio chunks."""
        loop = params.loop
        current_text_chunk_accumulated_tokens = None # Accumulates tokens for the current text chunk
        cache_source = torch.zeros(1, 1, 0).to(self.device) # Initialize cache_source for S3Gen streaming
        last_text_chunk_num = -1
        eos_token = torch.tensor([self.tts.t3.hp.stop_text_token]).unsqueeze(0).to(self.device)
        previous_audio_chunk = None  # Stores the remainder of the last processed chunk for the next cross-fade
        is_first_audio_chunk_sent = False # Flag to track if the very first audio chunk has been sent
        previous_length = 0 # Tracks audio length for 'full' overlap strategy

        try:
            while True:
                queue_item = await speech_token_queue.get()
                if queue_item is None:
                    # End of stream. No more audio to process.
                    # The last chunk should have already been sent by the main loop.
                    break

                token_chunk, text_chunk_num, slice_num, is_first_slice, is_last_slice, is_first_text_chunk, is_last_text_chunk, event = queue_item

                # Wait for the T3 slice to be ready before processing with S3Gen
                s3gen_stream.wait_event(event)
                log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                log.info(
                    f"{log_prefix}[{params.request_id}] S3Gen: Starting inference for slice {slice_num} "
                    f"(from text chunk {text_chunk_num}/{params.text_chunk_count})"
                )
                # Reset state for new text chunks
                if text_chunk_num != last_text_chunk_num:
                    current_text_chunk_accumulated_tokens = None # Reset for new text chunk
                    cache_source = torch.zeros(1, 1, 0).to(self.device) # Reset cache_source for new text chunk
                    last_text_chunk_num = text_chunk_num
                    previous_length = 0 # Reset for new text chunk

                # 1. Prepare tokens for S3Gen based on overlap method
                if params.chunk_overlap_strategy == "full":
                    if current_text_chunk_accumulated_tokens is None:
                        current_text_chunk_accumulated_tokens = token_chunk
                    else:
                        current_text_chunk_accumulated_tokens = torch.cat((current_text_chunk_accumulated_tokens, token_chunk), dim=1)
                    # The speech_tokens for the current inference is the accumulated tokens
                    speech_tokens_for_inference = current_text_chunk_accumulated_tokens
                else:  # zero overlap
                    speech_tokens_for_inference = token_chunk


                # Apply EOS token only for the last slice of the current text chunk
                if is_last_slice: # Apply EOS token at the end of each text chunk's last slice
                    speech_tokens_with_eos = torch.cat([speech_tokens_for_inference, eos_token], dim=1)
                    speech_tokens_for_inference = speech_tokens_with_eos[0]

                # with token filtering and were not necessary for the current streaming logic.
                speech_tokens_for_inference = drop_invalid_tokens(speech_tokens_for_inference)
                speech_tokens_for_inference = speech_tokens_for_inference[speech_tokens_for_inference < 6561]

                # If, after filtering, we have no valid tokens, skip this slice entirely.
                if speech_tokens_for_inference.shape[-1] == 0:
                    log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                    log.warning(f"{log_prefix}[{params.request_id}] Skipping a slice because it contained no valid tokens after filtering.")
                    speech_token_queue.task_done()
                    continue

                if speech_tokens_for_inference.shape[-1] < 3:
                    padding_needed = 3 - speech_tokens_for_inference.shape[-1]
                    speech_tokens_for_inference = F.pad(speech_tokens_for_inference, (0, padding_needed), "constant", 0)

                # 2. S3Gen Inference (blocking)
                inference_kwargs = {
                    "speech_tokens": speech_tokens_for_inference,
                    "ref_dict": params.conds.gen, # Always use the initial ref_dict
                }
                if params.chunk_overlap_strategy == "full":
                    inference_kwargs["cache_source"] = cache_source # Pass the cache_source for streaming

                partial_inference = functools.partial(
                    self.tts.s3gen.inference,
                    **inference_kwargs
                )
                s3gen_start_time = time.time()
                with torch.cuda.stream(s3gen_stream):
                    wav, new_cache_source = await loop.run_in_executor(None, partial_inference)
                s3gen_inference_time = time.time() - s3gen_start_time
                log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                log.info(f"{log_prefix}[{params.request_id}] S3Gen: Inference for slice {slice_num} took {s3gen_inference_time:.4f}s")
                current_audio_chunk = wav.squeeze(0).detach()

                # 3. Post-processing based on overlap method
                if params.chunk_overlap_strategy == "full":
                    # Update the cache_source for the next iteration
                    cache_source = new_cache_source
                    # Trim the repeated audio from the beginning of the chunk
                    new_audio_length = current_audio_chunk.shape[0]
                    if not is_first_slice:
                        current_audio_chunk = current_audio_chunk[previous_length:]
                    previous_length = new_audio_length

                # 4. Audio Processing (Trimming, Crossfading)
                # Calculate samples to remove based on milliseconds
                leading_samples_to_remove = (params.remove_leading_milliseconds * self.sr) // 1000
                trailing_samples_to_remove = (params.remove_trailing_milliseconds * self.sr) // 1000
                crossfade_samples = (params.crossfade_duration_milliseconds * self.sr) // 1000

                # --- Trimming Logic ---
                # Trim leading silence only from the very first chunk of the entire stream
                if is_first_text_chunk and is_first_slice:
                    if leading_samples_to_remove > 0 and current_audio_chunk.shape[0] > leading_samples_to_remove:
                        current_audio_chunk = current_audio_chunk[leading_samples_to_remove:]

                # Trim trailing silence only from the very last chunk of the entire stream
                if is_last_text_chunk and is_last_slice:
                    if trailing_samples_to_remove > 0 and current_audio_chunk.shape[0] > trailing_samples_to_remove:
                        current_audio_chunk = current_audio_chunk[:-trailing_samples_to_remove]


                # --- Crossfading Logic ---
                if previous_audio_chunk is not None and crossfade_samples > 0:
                    # Ensure we have enough audio data for a crossfade
                    if previous_audio_chunk.shape[0] > crossfade_samples and current_audio_chunk.shape[0] > crossfade_samples:
                        # Create the fade-out envelope for the previous chunk
                        fade_out = torch.linspace(1, 0, crossfade_samples).to(self.device)
                        # Create the fade-in envelope for the current chunk
                        fade_in = torch.linspace(0, 1, crossfade_samples).to(self.device)

                        # Apply the fade-out to the end of the previous chunk
                        previous_audio_chunk[-crossfade_samples:] *= fade_out
                        # Apply the fade-in to the beginning of the current chunk
                        current_audio_chunk[:crossfade_samples] *= fade_in

                        # Overlap and add the crossfaded sections
                        crossfaded_region = previous_audio_chunk[-crossfade_samples:] + current_audio_chunk[:crossfade_samples]

                        # Combine the parts: main part of previous, crossfaded region, main part of current
                        processed_chunk = torch.cat([
                            previous_audio_chunk[:-crossfade_samples],
                            crossfaded_region,
                            current_audio_chunk[crossfade_samples:]
                        ])
                    else:
                        # If chunks are too short for crossfading, just concatenate them
                        processed_chunk = torch.cat([previous_audio_chunk, current_audio_chunk])
                else:
                    processed_chunk = current_audio_chunk


                # --- Buffering for next crossfade ---
                # For all but the last slice, save the end of the processed chunk for the next crossfade
                if not is_last_slice:
                    if processed_chunk.shape[0] > crossfade_samples:
                        # The part to send immediately is everything except the crossfade tail
                        chunk_to_send = processed_chunk[:-crossfade_samples]
                        # The part to save for the next iteration is the crossfade tail
                        previous_audio_chunk = processed_chunk[-crossfade_samples:]
                    else:
                        # If the chunk is shorter than the crossfade duration, send nothing and buffer the whole chunk
                        chunk_to_send = None
                        previous_audio_chunk = processed_chunk
                else:
                    # This is the last slice of a text chunk. Send the whole thing.
                    chunk_to_send = processed_chunk
                    previous_audio_chunk = None # Reset for the next text chunk


                # 5. Queue the processed audio chunk for output
                if chunk_to_send is not None and chunk_to_send.shape[0] > 0:
                    pcm_data = await self.audio_processor.to_pcm(chunk_to_send, loop)
                    await pcm_chunk_queue.put(pcm_data)
                    log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                    log.info(f"{log_prefix}[{params.request_id}] S3Gen: Queued audio chunk of size {len(pcm_data)} bytes.")

                speech_token_queue.task_done()

            # After the loop, if there's any remaining audio in previous_audio_chunk, send it.
            # This happens for the very last chunk of the entire request.
            if previous_audio_chunk is not None and previous_audio_chunk.shape[0] > 0:
                pcm_data = await self.audio_processor.to_pcm(previous_audio_chunk, loop)
                await pcm_chunk_queue.put(pcm_data)
                log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
                log.info(f"{log_prefix}[{params.request_id}] S3Gen: Queued final remaining audio chunk.")

        except Exception as e:
            log_prefix = f"[GPU {self.gpu_id}]" if self.device != "cpu" else "[CPU]"
            log.error(f"{log_prefix}[{params.request_id}] Error in S3Gen consumer task: {e}", exc_info=True)
        finally:
            await pcm_chunk_queue.put(None) # Signal end of all audio


    async def stream(
        self,
        text: str,
        output_format: str,
        voice_id: Optional[str] = None,
        voice_exaggeration_factor: float = 0.5,
        cfg_guidance_weight: float = 2.0,
        synthesis_temperature: float = 0.9,
        text_processing_chunk_size: int = 120,
        audio_tokens_per_slice: int = 100,
        remove_trailing_milliseconds: int = 0,
        remove_leading_milliseconds: int = 0,
        chunk_overlap_strategy: Literal["zero", "full"] = "zero",
        crossfade_duration_milliseconds: int = 20,
        start_time: float = 0.0,
        request_id: str = "N/A",
        request: Optional[object] = None
    ) -> AsyncGenerator[bytes, None]:
        """Streams synthesized audio in the specified format."""
        if self._initialization_state != InitializationState.READY:
            raise RuntimeError(f"TTS Engine on GPU {self.gpu_id} is not ready. Status: {self._initialization_state.value}")

        loop = asyncio.get_running_loop()
        if request and hasattr(request.app.state, 'process_pool'):
            loop.process_pool = request.app.state.process_pool
        first_chunk_generated = False
        time_to_first_chunk = 0.0

        # 1. Get Voice Conditionals
        audio_prompt_path = self.voice_manager.get_voice_path(voice_id) if voice_id else None
        conds = await self._prepare_and_get_conds(audio_prompt_path, voice_exaggeration_factor, loop, request_id)

        # 2. Text Processing
        text_chunks = split_text_into_chunks(text, text_processing_chunk_size)
        if not text_chunks:
            yield b''
            return

        # 3. Setup Queues and CUDA Streams
        speech_token_queue = asyncio.Queue(maxsize=tts_config.SPEECH_TOKEN_QUEUE_MAX_SIZE)
        pcm_chunk_queue = asyncio.Queue(maxsize=tts_config.PCM_CHUNK_QUEUE_MAX_SIZE)
        t3_stream = torch.cuda.Stream() if self.device.startswith('cuda') else None
        s3gen_stream = torch.cuda.Stream() if self.device.startswith('cuda') else None

        # 4. Create Synthesis Parameters
        synthesis_params = SynthesisParams(
            text_tokens=None, # This will be set per-chunk in the producer
            conds=conds,
            cfg_guidance_weight=cfg_guidance_weight,
            synthesis_temperature=synthesis_temperature,
            text_processing_chunk_size=text_processing_chunk_size,
            audio_tokens_per_slice=audio_tokens_per_slice,
            remove_trailing_milliseconds=remove_trailing_milliseconds,
            remove_leading_milliseconds=remove_leading_milliseconds,
            chunk_overlap_strategy=chunk_overlap_strategy,
            crossfade_duration_milliseconds=crossfade_duration_milliseconds,
            loop=loop,
            text_chunk_count=len(text_chunks),
            request_id=request_id
        )

        # 5. Start Producer and Consumer Tasks
        producer_task = asyncio.create_task(
            self._t3_producer_task(text_chunks, speech_token_queue, synthesis_params, t3_stream, s3gen_stream)
        )
        consumer_task = asyncio.create_task(
            self._s3gen_consumer_task(speech_token_queue, pcm_chunk_queue, synthesis_params, s3gen_stream)
        )

        # 6. Setup Audio Encoder
        encoder = AudioEncoder(output_format, self.sr, request=request)

        async def pcm_generator():
            """Generator that yields PCM chunks from the queue."""
            while True:
                chunk = await pcm_chunk_queue.get()
                if chunk is None:
                    break
                yield chunk
                pcm_chunk_queue.task_done()

        # 7. Stream the output
        try:
            async for audio_chunk in encoder.encode(pcm_generator()):
                if not first_chunk_generated:
                    time_to_first_chunk = time.time() - start_time
                    log.info(f"[{request_id}] Time to first audio chunk: {time_to_first_chunk:.4f}s")
                    first_chunk_generated = True
                yield audio_chunk
        finally:
            # Cleanup: Cancel tasks and clear tensors
            producer_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(producer_task, consumer_task, return_exceptions=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class TTSEngineManager:
    """Manages multiple TTS engine instances, distributing load across available GPUs."""

    def __init__(self, num_gpus: int):
        """
        Initializes the manager and creates a TTS engine for each GPU.
        """
        self.num_gpus = num_gpus
        if self.num_gpus > 0:
            self.engines: List[TextToSpeechEngine] = [TextToSpeechEngine(gpu_id=i) for i in range(num_gpus)]
        else:
            # Fallback to a single CPU engine if no GPUs are detected
            self.engines: List[TextToSpeechEngine] = [TextToSpeechEngine(gpu_id=0)]
        self._lock = asyncio.Lock()
        self._next_engine_index = 0

    async def ainit(self):
        """
        Asynchronously initializes all managed TTS engines.
        """
        log.info(f"Initializing {len(self.engines)} TTS engine(s)...")
        init_tasks = [engine.ainit() for engine in self.engines]
        await asyncio.gather(*init_tasks)
        log.info("All TTS engines initialized.")

    async def get_engine(self) -> TextToSpeechEngine:
        """
        Selects the TTS engine with the fewest active requests.
        This ensures that load is distributed to the least busy engine.
        """
        if not self.engines:
            raise RuntimeError("No TTS engines available.")

        # Find the engine with the most available semaphore slots (least busy)
        # The semaphore's internal value is the number of free slots.
        best_engine = max(self.engines, key=lambda e: e.tts_semaphore._value)

        log.info(f"Selected least busy TTS engine on GPU {best_engine.gpu_id} with {best_engine.tts_semaphore._value} free slots.")
        return best_engine

    def get_all_engines(self) -> List[TextToSpeechEngine]:
        """Returns all engine instances."""
        return self.engines

    def get_status(self) -> dict:
        """
        Returns the initialization status of all managed engines.
        """
        return {f"gpu_{engine.gpu_id}": engine.get_initialization_status() for engine in self.engines}
