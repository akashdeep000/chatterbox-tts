"""
Main Text-to-Speech (TTS) engine module for production with enhanced streaming.

This module defines the primary `TextToSpeechEngine` class, which orchestrates
the TTS process. It integrates model loading, voice conditioning, caching,
and audio generation with performance optimizations.
"""

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Generator, AsyncGenerator, AsyncIterator
import io
import gc
import asyncio
import time
import functools
from enum import Enum

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
    # next_chunk_event: asyncio.Event # Removed for true streaming


class _AudioProcessor:
    """Handles audio processing tasks like WAV header creation and PCM conversion."""

    # TODO: Remove this method once we can stream the WAV header
    # @staticmethod
    # def create_wav_header(sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    #     """Creates a WAV file header for streaming."""
    #     data_size = 0xFFFFFFFF  # Set to 0xFFFFFFFF for streaming to indicate unknown size
    #     header = io.BytesIO()
    #     header.write(b'RIFF')
    #     header.write((data_size).to_bytes(4, 'little')) # Set to data_size (0xFFFFFFFF) for streaming
    #     header.write(b'WAVEfmt ')
    #     header.write((16).to_bytes(4, 'little'))
    #     header.write((1).to_bytes(2, 'little'))
    #     header.write(channels.to_bytes(2, 'little'))
    #     header.write(sample_rate.to_bytes(4, 'little'))
    #     header.write((sample_rate * channels * sample_width).to_bytes(4, 'little'))
    #     header.write((channels * sample_width).to_bytes(2, 'little'))
    #     header.write((sample_width * 8).to_bytes(2, 'little'))
    #     header.write(b'data')
    #     header.write(data_size.to_bytes(4, 'little'))
    #     header.seek(0)
    #     return header.read()

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
        return await loop.run_in_executor(None, _blocking_conversion)


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
        Initializes the TTS engine, setting up basic attributes.
        Model loading and device setup are handled asynchronously by `ainit`.
        """
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
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


            log.info(f"Initializing Chatterbox TTS model...")
            log.info(f"Device: {self.device}")
            log.info(f"Model path: {settings.MODEL_PATH}")

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
            log.info(f"✓ Model initialized successfully on {self.device}")
            return self.tts

        except Exception as e:
            self._initialization_state = InitializationState.ERROR
            self._initialization_error = str(e)
            self._initialization_progress = f"Failed: {str(e)}"
            log.error(f"✗ Failed to initialize model: {e}")
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

    async def _prepare_and_get_conds(self, audio_prompt_path: Optional[str], voice_exaggeration_factor: float, loop: asyncio.AbstractEventLoop) -> Conditionals:
        """Prepares and retrieves the appropriate conditionals."""
        voice_id = Path(audio_prompt_path).name if audio_prompt_path else "default"
        if audio_prompt_path and voice_id not in self.voice_cache:
            log.info(f"Voice '{voice_id}' not in cache. Preparing new conditionals...")
            await loop.run_in_executor(None, self.prepare_conditionals, audio_prompt_path, voice_exaggeration_factor)
            log.info(f"Finished preparing conditionals for '{voice_id}'.")
        elif audio_prompt_path:
            log.info(f"Using cached conditionals for voice '{voice_id}'.")

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
        stream: torch.cuda.Stream,
    ):
        """Producer task for T3 model. Generates speech tokens and puts them into a queue."""
        num_chunks = len(text_chunks)
        loop = params.loop
        try:
            for i, chunk in enumerate(text_chunks):
                is_first_text_chunk = (i == 0)
                is_last_text_chunk = (i == num_chunks - 1)

                log.info(f"T3: Processing text chunk {i+1}/{num_chunks}")
                # 1. Text to Tokens
                text_tokens = await loop.run_in_executor(
                    None, self.tts.tokenizer.text_to_tokens, chunk
                )
                # Move text_tokens to the correct device
                text_tokens = text_tokens.to(self.device)

                if params.cfg_guidance_weight > 0.0:
                    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
                sot, eot = self.tts.t3.hp.start_text_token, self.tts.t3.hp.stop_text_token
                text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)

                # 2. T3 Inference (blocking)
                log.info(f"T3: Starting inference for chunk {i+1}/{num_chunks}")
                # 3. T3 Inference (non-blocking)
                sync_token_generator = await loop.run_in_executor(
                    None,
                    self._blocking_t3_inference,
                    params.conds.t3,
                    text_tokens,
                    params.synthesis_temperature,
                    params.cfg_guidance_weight,
                    stream,
                )

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
                        await speech_token_queue.put(
                            (predicted_tokens, i + 1, slice_idx, is_first_slice, False, is_first_text_chunk, is_last_text_chunk)
                        )
                    elif generator_exhausted and current_slice:
                        # If no more tokens are coming, send full current_slice to queue
                        slice_idx += 1
                        is_first_slice = (slice_idx == 1)
                        # Concatenate all predicted tokens along the sequence dimension.
                        predicted_tokens = torch.cat(current_slice, dim=1)  # shape: (B, num_tokens)
                        await speech_token_queue.put(
                            (predicted_tokens, i + 1, slice_idx, is_first_slice, True, is_first_text_chunk, is_last_text_chunk) # shape: (B, num_tokens)
                        )
                        current_slice = [] # Clear current_slice after sending
                        break # All tokens sent for this chunk

                    # Break condition: if generator is exhausted and all buffered tokens have been sent
                    if generator_exhausted and not current_slice:
                        break

                log.info(f"T3: Finished inference for chunk {i+1}/{num_chunks}, produced {slice_idx} slices.")

        except Exception as e:
            log.error(f"Error in T3 producer task: {e}", exc_info=True)
        finally:
            await speech_token_queue.put(None) # Signal end of production

    async def _s3gen_consumer_task(
        self,
        speech_token_queue: asyncio.Queue,
        pcm_chunk_queue: asyncio.Queue,
        params: SynthesisParams,
        stream: torch.cuda.Stream,
    ):
        """Consumer task for S3Gen model. Converts speech tokens to audio chunks."""
        loop = params.loop
        current_text_chunk_accumulated_tokens = None # Accumulates tokens for the current text chunk
        cache_source = torch.zeros(1, 1, 0).to(self.device) # Initialize cache_source for S3Gen streaming
        last_text_chunk_num = -1
        eos_token = torch.tensor([self.tts.t3.hp.stop_text_token]).unsqueeze(0).to(self.device)
        previous_audio_chunk = None  # Stores the remainder of the last processed chunk for the next cross-fade
        is_first_audio_chunk_sent = False # Flag to track if the very first audio chunk has been sent

        try:
            while True:
                queue_item = await speech_token_queue.get()
                if queue_item is None:
                    # End of stream. No more audio to process.
                    # The last chunk should have already been sent by the main loop.
                    break

                token_chunk, text_chunk_num, slice_num, is_first_slice, is_last_slice, is_first_text_chunk, is_last_text_chunk = queue_item
                log.info(
                    f"S3Gen: Starting inference for slice {slice_num} "
                    f"(from text chunk {text_chunk_num}/{params.text_chunk_count})"
                )
                # Reset state for new text chunks
                if text_chunk_num != last_text_chunk_num:
                    current_text_chunk_accumulated_tokens = None # Reset for new text chunk
                    cache_source = torch.zeros(1, 1, 0).to(self.device) # Reset cache_source for new text chunk
                    last_text_chunk_num = text_chunk_num

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
                    log.warning("Skipping a slice because it contained no valid tokens after filtering.")
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
                with torch.cuda.stream(stream):
                    wav, new_cache_source = await loop.run_in_executor(None, partial_inference)
                current_audio_chunk = wav.squeeze(0).detach()

                if params.chunk_overlap_strategy == "full":
                    # Update cache_source for the next iteration within the same text chunk
                    cache_source = new_cache_source


                # 3. Post-processing based on overlap method
                if params.chunk_overlap_strategy == "full":
                    new_audio_length = current_audio_chunk.shape[0]
                    if not is_first_slice:
                        current_audio_chunk = current_audio_chunk[previous_length:]
                    previous_length = new_audio_length

                # These trims should apply to all methods for the respective slices
                if params.remove_trailing_milliseconds > 0 and is_last_slice:
                    trim_samples = int(self.sr * params.remove_trailing_milliseconds / 1000)
                    current_audio_chunk = current_audio_chunk[:-trim_samples]
                if params.remove_leading_milliseconds > 0 and is_first_slice:
                    trim_samples_start = int(self.sr * params.remove_leading_milliseconds / 1000)
                    current_audio_chunk = current_audio_chunk[trim_samples_start:]

                # 4. Cross-fading and Queueing
                output_to_send = None
                fade_len = int(self.sr * (params.crossfade_duration_milliseconds / 1000.0))

                if not is_first_audio_chunk_sent:
                    # --- First Chunk Logic ---
                    # Send the first chunk immediately without cross-fading.
                    # This is critical for reducing time-to-first-audio.
                    log.debug("First audio chunk generated. Sending immediately.")
                    # We send the main body and store the tail for the next fade.
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
                        params.crossfade_duration_milliseconds > 0
                        and fade_len > 0
                        and previous_audio_chunk is not None
                        and previous_audio_chunk.shape[0] == fade_len # Ensure prev is just the tail
                        and current_audio_chunk.shape[0] > fade_len
                    )

                    if can_fade:
                        # The head of the current chunk that will be faded in
                        current_head = current_audio_chunk[:fade_len]

                        # Equal power crossfade curves
                        t = torch.linspace(0, 1, fade_len, device=self.device)
                        fade_out = torch.cos(t * 0.5 * torch.pi)  # from 1 → 0
                        fade_in = torch.sin(t * 0.5 * torch.pi)   # from 0 → 1

                        # Apply fades: fade out the previous tail and fade in the current head
                        mixed_region = (previous_audio_chunk * fade_out) + (current_head * fade_in)

                        # The main body of the current chunk (the part after the fade)
                        # We also leave a tail for the *next* fade
                        current_main_body = current_audio_chunk[fade_len:-fade_len] if current_audio_chunk.shape[0] > fade_len * 2 else torch.tensor([], device=self.device)


                        # The output to send is the mixed region plus the main body of the current chunk.
                        output_to_send = torch.cat((mixed_region, current_main_body))

                        # The tail of the current chunk becomes the 'previous_audio_chunk' for the next iteration.
                        previous_audio_chunk = current_audio_chunk[-fade_len:]
                    else:
                        # --- No fade / Fallback ---
                        # Send the previous chunk (if any) and prepare the current one for the next iteration.
                        output_to_send = previous_audio_chunk
                        if fade_len > 0 and current_audio_chunk.shape[0] > fade_len:
                            # We can't fade, so just send the previous part and save the new tail
                            previous_audio_chunk = current_audio_chunk[-fade_len:]
                        else:
                            previous_audio_chunk = current_audio_chunk # Or store the whole thing if it's too short

                if output_to_send is not None and output_to_send.shape[0] > 0:
                    log.debug(f"Sending {output_to_send.shape[0]} samples to audio queue")
                    pcm_data = await self.audio_processor.to_pcm(output_to_send, loop)
                    await pcm_chunk_queue.put(pcm_data)

                speech_token_queue.task_done()
        except Exception as e:
            log.error(f"Error in S3Gen consumer task: {e}", exc_info=True)
        finally:
            # After the loop, send any final remaining audio chunk (likely the last tail).
            if previous_audio_chunk is not None and previous_audio_chunk.shape[0] > 0:
                log.debug(f"Sending the final remaining audio tail of {previous_audio_chunk.shape[0]} samples.")
                pcm_data = await self.audio_processor.to_pcm(previous_audio_chunk, loop)
                await pcm_chunk_queue.put(pcm_data)

            await pcm_chunk_queue.put(None)  # Signal end of audio stream

            log.info(
                    f"S3Gen: Finished inference for slice {slice_num} "
                    f"(from text chunk {text_chunk_num}/{params.text_chunk_count})"
                )

    async def stream(
        self,
        text: str,
        output_format: str = "wav",
        voice_id: Optional[str] = None,
        voice_exaggeration_factor: float = tts_config.VOICE_EXAGGERATION_FACTOR,
        cfg_guidance_weight: float = tts_config.CFG_GUIDANCE_WEIGHT,
        synthesis_temperature: float = tts_config.SYNTHESIS_TEMPERATURE,
        text_processing_chunk_size: Optional[int] = tts_config.TEXT_PROCESSING_CHUNK_SIZE,
        audio_tokens_per_slice: Optional[int] = tts_config.AUDIO_TOKENS_PER_SLICE,
        remove_trailing_milliseconds: int = tts_config.REMOVE_TRAILING_MILLISECONDS,
        remove_leading_milliseconds: int = tts_config.REMOVE_LEADING_MILLISECONDS,
        chunk_overlap_strategy: Literal["zero", "full"] = tts_config.CHUNK_OVERLAP_STRATEGY,
        crossfade_duration_milliseconds: int = tts_config.CROSSFADE_DURATION_MILLISECONDS,
        start_time: Optional[float] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Streams synthesized audio in the specified format."""
        loop = asyncio.get_running_loop()
        audio_prompt_path = self.voice_manager.get_voice_path(voice_id) if voice_id else None
        conds = await self._prepare_and_get_conds(audio_prompt_path, voice_exaggeration_factor, loop)
        yield_count = 0 # Counts the number of chunks yielded to the client

        text_chunks = split_text_into_chunks(text, text_processing_chunk_size)

        params = SynthesisParams(
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
        )

        speech_token_queue = asyncio.Queue(maxsize=tts_config.SPEECH_TOKEN_QUEUE_MAX_SIZE)
        pcm_chunk_queue = asyncio.Queue(maxsize=tts_config.PCM_CHUNK_QUEUE_MAX_SIZE)

        # Create streams for this specific request
        t3_stream = torch.cuda.Stream() if self.device == "cuda" else None
        s3gen_stream = torch.cuda.Stream() if self.device == "cuda" else None

        producer_task = loop.create_task(
            self._t3_producer_task(text_chunks, speech_token_queue, params, t3_stream)
        )
        consumer_task = loop.create_task(
            self._s3gen_consumer_task(speech_token_queue, pcm_chunk_queue, params, s3gen_stream)
        )

        async def pcm_generator():
            """Async generator to yield PCM chunks from the queue."""
            while True:
                chunk = await pcm_chunk_queue.get()
                if chunk is None:
                    break
                yield chunk
                pcm_chunk_queue.task_done()

        encoder = AudioEncoder(output_format, self.sr)
        encoded_stream = encoder.encode(pcm_generator())

        try:
            async for encoded_chunk in encoded_stream:
                yield_count += 1
                log.debug(f"Sending {len(encoded_chunk)} bytes to client (chunk {yield_count})")
                yield encoded_chunk

        finally:
            # This block now correctly waits for the client to be done, then cleans up.
            log.info("Client stream finished or disconnected. Cleaning up TTS tasks.")
            producer_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(producer_task, consumer_task, return_exceptions=True)
            log.info("TTS generation tasks cancelled and gathered.")
