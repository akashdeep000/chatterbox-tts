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
import concurrent.futures
import contextlib

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

from chatterbox.models.t3 import T3
from chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens
from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts import ChatterboxTTS as OriginalChatterboxTTS

# Local application/library specific imports
from src.utils import safe_delete_tensors
from src.config import settings, tts_config
from src.voice_manager import VoiceManager
from src.text_processing import split_text_into_chunks
from src.logging_config import log
from .audio_encoding import AudioEncoder

async def async_generator_wrapper(sync_generator: Generator, executor: concurrent.futures.Executor, batch_size: int = 32):
    """
    Wraps a synchronous generator to make it asynchronously iterable.
    Items are fetched in batches using a dedicated executor to reduce overhead and prevent blocking.
    """
    _END_OF_GENERATOR = object() # Local sentinel to signal generator exhaustion

    def _get_next_batch_safe():
        """Synchronous helper to get the next batch of items or signal exhaustion."""
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(sync_generator))
        except StopIteration:
            # This is expected when the generator is exhausted.
            # The batch might contain some remaining items.
            pass

        if not batch:
            return _END_OF_GENERATOR
        return batch

    loop = asyncio.get_running_loop()
    while True:
        # Run the synchronous helper in the provided dedicated executor
        item_batch = await loop.run_in_executor(executor, _get_next_batch_safe)
        if item_batch is _END_OF_GENERATOR:
            break

        # Yield each item from the fetched batch
        for item in item_batch:
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
    async def to_pcm(audio_tensor: torch.Tensor, loop: asyncio.AbstractEventLoop, executor: concurrent.futures.ThreadPoolExecutor) -> bytes:
        """
        Converts an audio tensor to PCM byte data asynchronously to avoid blocking.
        The tensor is moved to the CPU and converted inside a dedicated thread pool executor.
        """
        def _blocking_conversion():
            """The synchronous part of the conversion."""
            audio_tensor_clamped = torch.clamp(audio_tensor, -1.0, 1.0).cpu()
            audio_tensor_int = (audio_tensor_clamped * 32767).to(torch.int16)
            pcm_data = audio_tensor_int.numpy().tobytes()
            safe_delete_tensors(audio_tensor, audio_tensor_clamped, audio_tensor_int)
            return pcm_data

        # Offload the blocking CPU operations to the dedicated executor
        return await loop.run_in_executor(executor, _blocking_conversion)


class TextToSpeechEngine:
    """
    The main engine for Text-to-Speech synthesis.
    This class manages the entire TTS pipeline, including model loading,
    voice conditioning, audio generation, and streaming.
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, device: str):
        """
        Initializes the TTS engine for a single worker process.
        """
        self.device = device
        self.gpu_id = int(device.split(":")[-1]) if "cuda" in device else -1 # -1 for CPU
        self.tts = None # Will be loaded during async initialization
        self.voice_manager = VoiceManager()
        self.voice_cache: dict[str, Conditionals] = {}
        self._cached_audio_prompt_path: Optional[str] = None
        self.sr = S3GEN_SR # Default, will be updated after model loads
        self.audio_processor = _AudioProcessor()
        self._initialization_state: InitializationState = InitializationState.NOT_STARTED
        self._initialization_progress: str = ""
        self._initialization_error: Optional[str] = None
        self.tts_semaphore = asyncio.Semaphore(settings.CONCURRENT_REQUESTS_PER_WORKER)

        # Each worker process has its own simple, un-shared executors.
        self.pcm_conversion_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.CONCURRENT_REQUESTS_PER_WORKER)
        self.text_processing_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.CONCURRENT_REQUESTS_PER_WORKER)
        self.text_tokenization_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.CONCURRENT_REQUESTS_PER_WORKER)
        self.t3_producer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.CONCURRENT_REQUESTS_PER_WORKER)
        self.s3gen_producer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.CONCURRENT_REQUESTS_PER_WORKER)
        self.voice_conditioning_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.CONCURRENT_REQUESTS_PER_WORKER)

        self.fade_in_curve = None
        self.fade_out_curve = None

    def shutdown(self):
        """Gracefully shuts down the engine's internal thread pools."""
        log.info("Shutting down internal executors...")
        self.pcm_conversion_executor.shutdown(wait=True)
        self.text_processing_executor.shutdown(wait=True)
        self.text_tokenization_executor.shutdown(wait=True)
        self.t3_producer_executor.shutdown(wait=True)
        self.s3gen_producer_executor.shutdown(wait=True)
        self.voice_conditioning_executor.shutdown(wait=True)
        log.info("Internal executors shut down.")

    async def ainit(self):
        """
        Asynchronously initializes the TTS engine, loads models, and sets the computation device.
        It automatically detects and uses CUDA or MPS if available, otherwise falls back to CPU.
        """
        try:
            self._initialization_state = InitializationState.INITIALIZING
            self._initialization_progress = "Validating configuration..."

            log.info("Initializing Chatterbox TTS model...")
            log.info(f"Device: {self.device}, Model path: {settings.MODEL_PATH}")
            if self.device == "cpu":
                log.warning("WARNING: No CUDA or MPS device found. Falling back to CPU. Performance will be significantly degraded.")


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
                self.text_processing_executor,
                lambda: OriginalChatterboxTTS.from_local(
                    settings.MODEL_PATH,
                    device=self.device
                )
            )
            self.sr = self.tts.sr # Set sample rate from the loaded model

            # Compile the T3 model for performance, with a fallback for safety.
            log.info("Attempting to compile T3 model with torch.compile...")
            try:
                compiled_t3 = torch.compile(self.tts.t3, mode="reduce-overhead", fullgraph=True)
                self.tts.t3 = compiled_t3
                # Verify that the model is now a compiled module
                if "OptimizedModule" in str(type(self.tts.t3)):
                    log.info("T3 model successfully compiled.")
                else:
                    log.warning(f"T3 model compilation did not return an OptimizedModule. Type is {type(self.tts.t3)}")
            except Exception as e:
                log.warning(f"T3 model compilation failed: {e}. Falling back to the original model.")

            # Warm-up run to pay the compilation cost upfront
            log.info("Performing warm-up run for compiled T3 model...")
            try:
                # Use default conditionals if available, otherwise this will raise an error
                conds = await self._prepare_and_get_conds(None, 0.5, loop, "warmup")

                # Create dummy text input
                warmup_text = "compiling"
                text_tokens = self.tts.tokenizer.text_to_tokens(warmup_text).to(self.device)
                sot, eot = self.tts.t3.hp.start_text_token, self.tts.t3.hp.stop_text_token
                text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)

                # Run a few steps of inference to trigger compilation
                gen = self.tts.t3.inference_stream(
                    t3_cond=conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=4, # Just need a few tokens
                    cfg_weight=0.0
                )
                # Capture a token from the T3 warm-up to use for the S3Gen warm-up
                warmup_speech_tokens = None
                for token in gen:
                    warmup_speech_tokens = token
                    break # We only need one token

                log.info("T3 model warm-up complete.")

                # Compile and warm-up the S3Gen model, with a fallback for safety.
                log.info("Attempting to compile S3Gen model with torch.compile...")
                try:
                    compiled_s3gen = torch.compile(self.tts.s3gen, mode="reduce-overhead", fullgraph=True)
                    self.tts.s3gen = compiled_s3gen
                    if "OptimizedModule" in str(type(self.tts.s3gen)):
                        log.info("S3Gen model successfully compiled.")
                    else:
                        log.warning(f"S3Gen model compilation did not return an OptimizedModule. Type is {type(self.tts.s3gen)}")
                except Exception as e:
                    log.warning(f"S3Gen model compilation failed: {e}. Falling back to the original model.")

                log.info("Performing warm-up run for compiled S3Gen model...")
                if warmup_speech_tokens is not None:
                    # Use the token from the T3 warm-up to warm up S3Gen
                    self.tts.s3gen.inference(
                        speech_tokens=warmup_speech_tokens,
                        ref_dict=conds.gen,
                        cache_source=torch.zeros(1, 1, 0).to(self.device)
                    )
                    log.info("S3Gen model warm-up complete.")
                else:
                    log.warning("S3Gen model warm-up skipped: no tokens from T3 warm-up.")

            except Exception as e:
                log.warning(f"Model warm-up failed: {e}. The first request may be slow.")

            self._initialization_state = InitializationState.READY
            self._initialization_progress = "Model ready"
            self._initialization_error = None
            log.info(f"✓ Model initialized successfully on {self.device}")
            return self.tts

        except Exception as e:
            self._initialization_state = InitializationState.ERROR
            self._initialization_error = str(e)
            self._initialization_progress = f"Failed: {str(e)}"
            log.error(f"✗ Failed to initialize model: {e}", exc_info=True)
            raise e

    def get_initialization_status(self) -> dict:
        """Returns the current initialization status of the TTS engine."""
        return {
            "state": self._initialization_state.value,
            "progress": self._initialization_progress,
            "error": self._initialization_error
        }

    def clear_voice_cache(self, voice_id: str):
        """Clears a specific voice from the cache."""
        if voice_id in self.voice_cache:
            del self.voice_cache[voice_id]
            log.info(f"Removed voice '{voice_id}' from cache.")
        else:
            log.warning(f"Attempted to clear non-existent voice '{voice_id}' from cache.")

    def prepare_conditionals(self, wav_fpath: str):
        """
        Prepares conditioning information from a reference audio file using the
        global voice_exaggeration_factor from the configuration.
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
            emotion_adv=tts_config.VOICE_EXAGGERATION_FACTOR * torch.ones(1, 1, 1),
        ).to(device=torch.device(self.device))

        voice_id = Path(wav_fpath).name
        self.voice_cache[voice_id] = Conditionals(t3_cond, s3gen_ref_dict)

    async def _prepare_and_get_conds(self, audio_prompt_path: Optional[str], loop: asyncio.AbstractEventLoop, request_id: str) -> Conditionals:
        """
        Prepares and retrieves the appropriate conditionals. Since voice_exaggeration_factor
        is now a global setting, we no longer need to check for per-request changes.
        """
        voice_id = Path(audio_prompt_path).name if audio_prompt_path else "default"
        if audio_prompt_path and voice_id not in self.voice_cache:
            log.info(f"[{request_id}] Voice '{voice_id}' not in cache. Preparing new conditionals...")
            await loop.run_in_executor(self.voice_conditioning_executor, self.prepare_conditionals, audio_prompt_path)
            log.info(f"[{request_id}] Finished preparing conditionals for '{voice_id}'.")
        elif audio_prompt_path:
            log.info(f"[{request_id}] Using cached conditionals for voice '{voice_id}'.")

        conds = self.voice_cache.get(voice_id)
        if conds is None:
            if self.tts.conds:
                conds = self.tts.conds
            else:
                raise ValueError("No audio prompt provided, and no default conditionals are loaded.")

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
        # Use the CUDA stream context only if a stream is provided (i.e., on a CUDA device)
        if stream:
            with torch.cuda.stream(stream):
                return self.tts.t3.inference_stream(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    max_new_tokens=1000,
                    temperature=synthesis_temperature,
                    cfg_weight=cfg_guidance_weight,
                )
        else:
            # For CPU, run without the stream context
            return self.tts.t3.inference_stream(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=synthesis_temperature,
                cfg_weight=cfg_guidance_weight,
            )

    async def _t3_producer_task(
        self,
        text_chunks: list,
        speech_token_queue: asyncio.Queue,
        gpu_audio_queue: asyncio.Queue,
        pcm_chunk_queue: asyncio.Queue,
        params: SynthesisParams,
        t3_stream: Optional[torch.cuda.Stream],
        s3gen_stream: Optional[torch.cuda.Stream],
    ):
        """Producer task for T3 model. Generates speech tokens and puts them into a queue."""
        num_chunks = len(text_chunks)
        loop = params.loop
        log_prefix = f"[{params.request_id}][T3_PRODUCER]"

        try:
            for i, chunk in enumerate(text_chunks):
                is_first_text_chunk = (i == 0)
                is_last_text_chunk = (i == num_chunks - 1)

                log.info(f"{log_prefix} Processing text chunk {i+1}/{num_chunks}. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                # 1. Text to Tokens (using ProcessPoolExecutor for true parallelism)
                tokenizer_start_time = time.time()
                text_tokens = await loop.run_in_executor(
                    self.text_tokenization_executor, self.tts.tokenizer.text_to_tokens, chunk
                )
                log.info(f"{log_prefix} Tokenizer took {time.time() - tokenizer_start_time:.4f}s")
                is_first_text_chunk = (i == 0)
                is_last_text_chunk = (i == num_chunks - 1)

                log.info(f"{log_prefix} Processing tokenized chunk {i+1}/{num_chunks}. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")

                # Move text_tokens to the correct device
                text_tokens = text_tokens.to(self.device)

                if params.cfg_guidance_weight > 0.0:
                    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
                sot, eot = self.tts.t3.hp.start_text_token, self.tts.t3.hp.stop_text_token
                text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)

                # 2. T3 Inference (blocking)
                log.info(f"{log_prefix} Starting inference for chunk {i+1}/{num_chunks}. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                # 3. T3 Inference (non-blocking, using the dedicated T3 executor)
                sync_token_generator = await loop.run_in_executor(
                    self.t3_producer_executor,
                    self._blocking_t3_inference,
                    params.conds.t3,
                    text_tokens,
                    params.synthesis_temperature,
                    params.cfg_guidance_weight,
                    t3_stream,
                )

                # 4. Stream individual tokens and accumulate into slices before queuing
                t3_start_time = time.time()
                current_slice = [] # array of single token [token shape: (B, 1)]
                slice_idx = 0

                # Define the look-ahead buffer size (20% of audio_tokens_per_slice or 10 tokens, whichever is larger)
                look_ahead_buffer_size = max(3, int(0.2 * params.audio_tokens_per_slice))
                # Create an iterator from the async generator
                token_iterator = async_generator_wrapper(sync_token_generator, self.t3_producer_executor, batch_size=(look_ahead_buffer_size + params.audio_tokens_per_slice)).__aiter__()
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
                        event = torch.cuda.Event() if t3_stream else None
                        if event:
                            event.record(t3_stream)
                        log.debug(f"{log_prefix} Queuing slice {slice_idx} ({predicted_tokens.shape[1]} tokens) from text chunk {i+1}/{num_chunks}. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
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
                        event = torch.cuda.Event() if t3_stream else None
                        if event:
                            event.record(t3_stream)
                        log.debug(f"{log_prefix} Queuing LAST slice {slice_idx} ({predicted_tokens.shape[1]} tokens) from text chunk {i+1}/{num_chunks}. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                        await speech_token_queue.put(
                            (predicted_tokens, i + 1, slice_idx, is_first_slice, True, is_first_text_chunk, is_last_text_chunk, event)
                        )
                        current_slice = [] # Clear current_slice after sending
                        break # All tokens sent for this chunk

                    # Break condition: if generator is exhausted and all buffered tokens have been sent
                    if generator_exhausted and not current_slice:
                        break

                # Yield control to the event loop to allow other tasks to run.
                await asyncio.sleep(0)

                t3_inference_time = time.time() - t3_start_time
                log.info(f"{log_prefix} Finished inference for chunk {i+1}/{num_chunks} ({slice_idx} slices) in {t3_inference_time:.4f}s. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")

        except Exception as e:
            log.error(f"{log_prefix} Error: {e}", exc_info=True)
        finally:
            log.info(f"{log_prefix} Finished. Signaling end of production.")
            await speech_token_queue.put(None) # Signal end of production

    def _blocking_s3gen_inference(self, speech_tokens_for_inference, ref_dict, cache_source, s3gen_stream):
        """Synchronous helper for S3Gen inference."""
        with torch.cuda.stream(s3gen_stream) if s3gen_stream else contextlib.nullcontext():
            return self.tts.s3gen.inference(
                speech_tokens=speech_tokens_for_inference,
                ref_dict=ref_dict,
                cache_source=cache_source
            )

    async def _s3gen_producer_task(
        self,
        speech_token_queue: asyncio.Queue,
        gpu_audio_queue: asyncio.Queue,
        pcm_chunk_queue: asyncio.Queue,
        params: SynthesisParams,
        s3gen_stream: Optional[torch.cuda.Stream],
    ):
        """Producer task for S3Gen model. Converts speech tokens to audio chunks, handling all GPU-side processing."""
        loop = params.loop
        log_prefix = f"[{params.request_id}][S3GEN_PRODUCER]"
        current_text_chunk_accumulated_tokens = None
        cache_source = torch.zeros(1, 1, 0).to(self.device)
        last_text_chunk_num = -1
        eos_token = torch.tensor([self.tts.t3.hp.stop_text_token]).unsqueeze(0).to(self.device)
        previous_audio_chunk = None
        is_first_audio_chunk_sent = False

        try:
            while True:
                start_time = time.time()
                log.debug(f"{log_prefix} Waiting for speech tokens. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                queue_item = await speech_token_queue.get()
                wait_time = time.time() - start_time
                log.debug(f"{log_prefix} Waited for speech tokens for {wait_time:.4f}s. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                if queue_item is None:
                    break

                token_chunk, text_chunk_num, slice_num, is_first_slice, is_last_slice, is_first_text_chunk, is_last_text_chunk, event = queue_item

                if s3gen_stream and event:
                    s3gen_stream.wait_event(event)

                log.info(f"{log_prefix} Starting inference for slice {slice_num} (text chunk {text_chunk_num}/{params.text_chunk_count}). Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")

                if text_chunk_num != last_text_chunk_num:
                    log.debug(f"{log_prefix} New text chunk detected. Resetting S3Gen cache.")
                    current_text_chunk_accumulated_tokens = None
                    cache_source = torch.zeros(1, 1, 0).to(self.device)
                    last_text_chunk_num = text_chunk_num
                    previous_length = 0

                if params.chunk_overlap_strategy == "full":
                    current_text_chunk_accumulated_tokens = token_chunk if current_text_chunk_accumulated_tokens is None else torch.cat((current_text_chunk_accumulated_tokens, token_chunk), dim=1)
                    speech_tokens_for_inference = current_text_chunk_accumulated_tokens
                else:
                    speech_tokens_for_inference = token_chunk

                # 1. Prepare tokens for S3Gen
                if is_last_slice: # Apply EOS token at the end of each text chunk's last slice
                    speech_tokens_with_eos = torch.cat([speech_tokens_for_inference, eos_token], dim=1)
                    speech_tokens_for_inference = speech_tokens_with_eos[0]

                # with token filtering and were not necessary for the current streaming logic.
                speech_tokens_for_inference = drop_invalid_tokens(speech_tokens_for_inference)
                speech_tokens_for_inference = speech_tokens_for_inference[speech_tokens_for_inference < 6561]

                if speech_tokens_for_inference.shape[-1] == 0:
                    log.debug(f"[{params.request_id}] Skipping a slice because it contained no valid tokens after filtering.")
                    speech_token_queue.task_done()
                    continue

                if speech_tokens_for_inference.shape[-1] < 3:
                    padding_needed = 3 - speech_tokens_for_inference.shape[-1]
                    speech_tokens_for_inference = F.pad(speech_tokens_for_inference, (0, padding_needed), "constant", 0)

                # 2. S3Gen Inference
                s3gen_start_time = time.time()
                wav, new_cache_source = await loop.run_in_executor(
                    self.s3gen_producer_executor,
                    self._blocking_s3gen_inference,
                    speech_tokens_for_inference,
                    params.conds.gen,
                    cache_source,
                    s3gen_stream
                )
                s3gen_inference_time = time.time() - s3gen_start_time
                log.info(f"{log_prefix} Inference for slice {slice_num} took {s3gen_inference_time:.4f}s. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                current_audio_chunk = wav.squeeze(0).detach()

                # 3. Overlap Handling
                if params.chunk_overlap_strategy == "full":
                    cache_source = new_cache_source
                    new_audio_length = current_audio_chunk.shape[0]
                    if not is_first_slice:
                        current_audio_chunk = current_audio_chunk[previous_length:]
                    previous_length = new_audio_length

                # 4. GPU-side Trimming
                leading_samples_to_remove = (params.remove_leading_milliseconds * self.sr) // 1000
                trailing_samples_to_remove = (params.remove_trailing_milliseconds * self.sr) // 1000
                if is_first_text_chunk and is_first_slice and leading_samples_to_remove > 0 and current_audio_chunk.shape[0] > leading_samples_to_remove:
                    current_audio_chunk = current_audio_chunk[leading_samples_to_remove:]
                if is_last_text_chunk and is_last_slice and trailing_samples_to_remove > 0 and current_audio_chunk.shape[0] > trailing_samples_to_remove:
                    current_audio_chunk = current_audio_chunk[:-trailing_samples_to_remove]

                # 4. Cross-fading and Queueing
                output_to_send = None

                fade_len = self.fade_in_curve.shape[0] if self.fade_in_curve is not None else 0

                if not is_first_audio_chunk_sent:
                    # --- First Chunk Logic ---
                    log.debug(f"{log_prefix} First audio chunk generated. Sending immediately.")
                    if fade_len > 0 and current_audio_chunk.shape[0] > fade_len:
                        output_to_send = current_audio_chunk[:-fade_len]
                        previous_audio_chunk = current_audio_chunk[-fade_len:]
                    else:
                        output_to_send = current_audio_chunk
                        previous_audio_chunk = None
                    is_first_audio_chunk_sent = True
                else:
                    # --- Standard Cross-fade Logic ---
                    can_fade = (
                        self.fade_in_curve is not None
                        and previous_audio_chunk is not None
                        and previous_audio_chunk.shape[0] == fade_len
                        and current_audio_chunk.shape[0] > fade_len
                    )

                    if can_fade:
                        current_head = current_audio_chunk[:fade_len]
                        mixed_region = (previous_audio_chunk * self.fade_out_curve) + (current_head * self.fade_in_curve)

                        current_main_body = current_audio_chunk[fade_len:-fade_len] if current_audio_chunk.shape[0] > fade_len * 2 else torch.tensor([], device=self.device)
                        output_to_send = torch.cat((mixed_region, current_main_body))
                        previous_audio_chunk = current_audio_chunk[-fade_len:]
                    else:
                        # --- No fade / Fallback ---
                        output_to_send = previous_audio_chunk
                        if fade_len > 0 and current_audio_chunk.shape[0] > fade_len:
                            previous_audio_chunk = current_audio_chunk[-fade_len:]
                        else:
                            previous_audio_chunk = current_audio_chunk

                if output_to_send is not None and output_to_send.shape[0] > 0:
                    log.debug(f"{log_prefix} Sending {output_to_send.shape[0]} samples to audio queue")
                    await gpu_audio_queue.put((output_to_send, text_chunk_num, slice_num, is_first_slice, is_last_slice, is_first_text_chunk, is_last_text_chunk))

                speech_token_queue.task_done()
        except Exception as e:
            log.error(f"{log_prefix} Error in S3Gen producer task: {e}", exc_info=True)
        finally:
            if previous_audio_chunk is not None and previous_audio_chunk.shape[0] > 0:
                log.debug(f"{log_prefix} Sending the final remaining audio tail of {previous_audio_chunk.shape[0]} samples.")
                await gpu_audio_queue.put((previous_audio_chunk, text_chunk_num, slice_num, is_first_slice, True, is_first_text_chunk, True))

            await gpu_audio_queue.put(None)

    async def _pcm_consumer_task(
        self,
        speech_token_queue: asyncio.Queue,
        gpu_audio_queue: asyncio.Queue,
        pcm_chunk_queue: asyncio.Queue,
        params: SynthesisParams,
    ):
        """Consumes audio tensors from the GPU queue and converts them to PCM data."""
        loop = params.loop
        log_prefix = f"[{params.request_id}][PCM_CONSUMER]"
        try:
            while True:
                start_time = time.time()
                log.debug(f"{log_prefix} Waiting for audio tensor. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                queue_item = await gpu_audio_queue.get()
                wait_time = time.time() - start_time
                log.debug(f"{log_prefix} Waited for audio tensor for {wait_time:.4f}s. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                if queue_item is None:
                    break
                audio_tensor, text_chunk_num, slice_num, is_first_slice, is_last_slice, is_first_text_chunk, is_last_text_chunk = queue_item
                start_time = time.time()
                pcm_data = await self.audio_processor.to_pcm(audio_tensor, loop, self.pcm_conversion_executor)
                conversion_time = time.time() - start_time
                log.info(f"{log_prefix} PCM conversion for slice {slice_num} (text chunk {text_chunk_num}/{params.text_chunk_count}) took {conversion_time:.4f}s. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                await pcm_chunk_queue.put(pcm_data)

        except Exception as e:
            log.error(f"{log_prefix} Error: {e}", exc_info=True)
        finally:
            log.info(f"{log_prefix} Finished. Signaling end of production.")
            await pcm_chunk_queue.put(None)


    async def stream(
        self,
        text: str,
        output_format: str,
        voice_id: Optional[str],
        cfg_guidance_weight: float,
        synthesis_temperature: float,
        text_processing_chunk_size: int,
        audio_tokens_per_slice: int,
        remove_trailing_milliseconds: int,
        remove_leading_milliseconds: int,
        chunk_overlap_strategy: Literal["zero", "full"],
        crossfade_duration_milliseconds: int,
        request_id: str
    ) -> AsyncGenerator[bytes, None]:
        """Streams synthesized audio in the specified format."""
        if self._initialization_state != InitializationState.READY:
            raise RuntimeError(f"TTS Engine on GPU {self.gpu_id} is not ready. Status: {self._initialization_state.value}")

        loop = asyncio.get_running_loop()
        first_chunk_generated = False
        time_to_first_chunk = 0.0
        start_time = time.time()

        # 1. Get Voice Conditionals
        audio_prompt_path = self.voice_manager.get_voice_path(voice_id) if voice_id else None
        conds = await self._prepare_and_get_conds(audio_prompt_path, loop, request_id)

        # 2. Text Processing
        text_chunks = await loop.run_in_executor(
            self.text_processing_executor,
            split_text_into_chunks,
            text,
            text_processing_chunk_size
        )
        if not text_chunks:
            yield b''
            return

        # 3. Setup Queues and CUDA Streams
        speech_token_queue = asyncio.Queue(maxsize=tts_config.SPEECH_TOKEN_QUEUE_MAX_SIZE)
        gpu_audio_queue = asyncio.Queue(maxsize=tts_config.PCM_CHUNK_QUEUE_MAX_SIZE)
        pcm_chunk_queue = asyncio.Queue(maxsize=tts_config.PCM_CHUNK_QUEUE_MAX_SIZE)
        is_cuda = self.device.startswith('cuda')
        t3_stream = torch.cuda.Stream() if is_cuda else None
        s3gen_stream = torch.cuda.Stream() if is_cuda else None

        # Pre-calculate fade curves to avoid repeated computation in the hot loop
        fade_len = int(self.sr * (crossfade_duration_milliseconds / 1000.0))
        if fade_len > 0:
            t = torch.linspace(0, 1, fade_len, device=self.device)
            self.fade_out_curve = torch.cos(t * 0.5 * torch.pi)
            self.fade_in_curve = torch.sin(t * 0.5 * torch.pi)
        else:
            self.fade_in_curve = None
            self.fade_out_curve = None

        # 4. Create Synthesis Parameters
        synthesis_params = SynthesisParams(
            text_tokens=None,
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
        t3_producer = asyncio.create_task(
            self._t3_producer_task(text_chunks, speech_token_queue, gpu_audio_queue, pcm_chunk_queue, synthesis_params, t3_stream, s3gen_stream)
        )
        s3gen_producer = asyncio.create_task(
            self._s3gen_producer_task(speech_token_queue, gpu_audio_queue, pcm_chunk_queue, synthesis_params, s3gen_stream)
        )
        pcm_consumer = asyncio.create_task(
            self._pcm_consumer_task(speech_token_queue, gpu_audio_queue, pcm_chunk_queue, synthesis_params)
        )

        # 6. Setup Audio Encoder
        log_prefix = f"[{request_id}]"

        encoder = AudioEncoder(
            output_format,
            self.sr,
            log_prefix=log_prefix
        )

        async def pcm_generator():
            """Generator that yields PCM chunks from the queue."""
            while True:
                log.debug(f"{log_prefix} Waiting for PCM chunk. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                chunk = await pcm_chunk_queue.get()
                wait_time = time.time() - start_time
                log.debug(f"{log_prefix} Waited for PCM chunk for {wait_time:.4f}s. Queues: speech_token_q:{speech_token_queue.qsize()}, gpu_audio_q:{gpu_audio_queue.qsize()}, pcm_chunk_q:{pcm_chunk_queue.qsize()}")
                if chunk is None:
                    log.debug(f"{log_prefix} PCM chunk queue is empty. Breaking.")
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
            log.info(f"{log_prefix} Cleaning up...")
            tasks = [t3_producer, s3gen_producer, pcm_consumer]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info(f"{log_prefix} Cleanup complete.")


