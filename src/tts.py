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
from enum import Enum


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
    next_chunk_event: asyncio.Event


class _AudioProcessor:
    """Handles audio processing tasks like WAV header creation and PCM conversion."""

    @staticmethod
    def create_wav_header(sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
        """Creates a WAV file header for streaming."""
        data_size = 0xFFFFFFFF  # Set to 0xFFFFFFFF for streaming to indicate unknown size
        header = io.BytesIO()
        header.write(b'RIFF')
        header.write((data_size).to_bytes(4, 'little')) # Set to data_size (0xFFFFFFFF) for streaming
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
        Initializes the TTS engine, setting up basic attributes.
        Model loading and device setup are handled asynchronously by `ainit`.
        """
        self.device = None # Will be set during async initialization
        self.t3_stream = None
        self.s3gen_stream = None
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

            self.t3_stream = None
            self.s3gen_stream = None
            if self.device == "cuda":
                self.t3_stream = torch.cuda.Stream()
                self.s3gen_stream = torch.cuda.Stream()

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
            # Directly return the generator from tts.t3.inference
            return self.tts.t3.inference(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                max_new_tokens=500, # Consider making this configurable if needed
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
                if i > 0:
                    # Wait for the signal from the consumer before starting the next chunk.
                    await params.next_chunk_event.wait()
                    params.next_chunk_event.clear()

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
                token_stream = await loop.run_in_executor(
                    None,
                    self._blocking_t3_inference,
                    params.conds.t3,
                    text_tokens,
                    params.synthesis_temperature,
                    params.cfg_guidance_weight,
                    stream,
                )

                # 4. Slice the buffered tokens and queue them with metadata
                # Explicitly create slices from the token stream.
                # This step is necessary to determine total_slices before queuing,
                # as total_slices is used for trigger_point calculation.
                slices = []
                current_slice = []
                for token_batch in token_stream: # token_stream is now a generator yielding batches
                    for token in token_batch.squeeze(0): # Iterate over individual tokens in the batch
                        current_slice.append(token)
                        if len(current_slice) >= params.audio_tokens_per_slice:
                            slices.append(current_slice)
                            current_slice = []

                if current_slice:
                    # Handle the last, potentially short, slice.
                    merge_threshold = max(10, int(params.audio_tokens_per_slice * 0.5))
                    if slices and len(current_slice) < merge_threshold:
                        log.info(
                            f"Merging a very short final slice (length {len(current_slice)}, "
                            f"threshold is {merge_threshold}) with the previous one."
                        )
                        slices[-1].extend(current_slice)
                    else:
                        slices.append(current_slice)

                total_slices = len(slices)
                # Define the trigger point for the consumer to signal back.
                # It's the lower of the second-to-last slice, but at least 1 or bases on audio_tokens_per_slice/text_processing_chunk_size
                trigger_point = max(1, min(total_slices - 1, int(total_slices * (params.audio_tokens_per_slice / params.text_processing_chunk_size))))

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
        stream: torch.cuda.Stream,
    ):
        """Consumer task for S3Gen model. Converts speech tokens to audio chunks."""
        loop = params.loop
        previous_length = 0
        accumulated_tokens = None # Initialize as None for tensor accumulation
        last_text_chunk_num = -1
        previous_audio_chunk = None
        # Move eos_token creation outside the loop
        eos_token = torch.tensor([self.tts.t3.hp.stop_text_token]).unsqueeze(0).to(self.device)

        try:
            while True:
                queue_item = await speech_token_queue.get()
                if queue_item is None:
                    # End of stream. No more audio to process.
                    # The last chunk should have already been sent by the main loop.
                    break

                token_chunk, text_chunk_num, slice_num, total_slices, trigger_point = queue_item
                log.info(
                    f"S3Gen: Starting inference for slice {slice_num}/{total_slices} "
                    f"(from text chunk {text_chunk_num}/{params.text_chunk_count})"
                )

                # Reset state for new text chunks
                if text_chunk_num != last_text_chunk_num:
                    accumulated_tokens = None # Reset to None for new text chunk
                    previous_length = 0
                    last_text_chunk_num = text_chunk_num

                # 1. Prepare tokens for S3Gen based on overlap method
                if params.chunk_overlap_strategy == "full":
                    if accumulated_tokens is None:
                        accumulated_tokens = token_chunk
                    else:
                        accumulated_tokens = torch.cat((accumulated_tokens, token_chunk), dim=0)
                    speech_tokens = accumulated_tokens.unsqueeze(0)
                else:  # zero overlap
                    speech_tokens = token_chunk.unsqueeze(0)
                # eos_token is already moved outside the loop
                tokens_with_eos = torch.cat([speech_tokens, eos_token], dim=1)
                speech_tokens = tokens_with_eos[0]
                speech_tokens = drop_invalid_tokens(speech_tokens)
                # Remove redundant .to(self.device) as speech_tokens should already be on device
                speech_tokens = speech_tokens[speech_tokens < 6561]

                # If, after filtering, we have no valid tokens, skip this slice entirely.
                if speech_tokens.shape[-1] == 0:
                    log.warning("Skipping a slice because it contained no valid tokens after filtering.")
                    speech_token_queue.task_done()
                    continue

                if speech_tokens.shape[-1] < 3:
                    padding_needed = 3 - speech_tokens.shape[-1]
                    speech_tokens = F.pad(speech_tokens, (0, padding_needed), "constant", 0)

                # 2. S3Gen Inference (blocking)
                partial_inference = functools.partial(
                    self.tts.s3gen.inference, speech_tokens=speech_tokens, ref_dict=params.conds.gen
                )
                with torch.cuda.stream(stream):
                    wav, _ = await loop.run_in_executor(None, partial_inference)
                current_audio_chunk = wav.squeeze(0).detach()

                # 3. Post-processing based on overlap method
                if params.chunk_overlap_strategy == "full":
                    new_audio_length = current_audio_chunk.shape[0]
                    if previous_length > 0:
                        current_audio_chunk = current_audio_chunk[previous_length:]
                    previous_length = new_audio_length

                # These trims should apply to all methods for the respective slices
                if params.remove_trailing_milliseconds > 0 and slice_num == total_slices:
                    trim_samples = int(self.sr * params.remove_trailing_milliseconds / 1000)
                    current_audio_chunk = current_audio_chunk[:-trim_samples]
                if params.remove_leading_milliseconds > 0 and slice_num == 1:
                    trim_samples_start = int(self.sr * params.remove_leading_milliseconds / 1000)
                    current_audio_chunk = current_audio_chunk[trim_samples_start:]

                # 4. Cross-fading and Queueing
                output_to_send = None
                if previous_audio_chunk is None:
                    # This is the very first chunk. Send it immediately.
                    output_to_send = current_audio_chunk
                    # For the next iteration, this current_audio_chunk becomes the 'previous' one.
                    previous_audio_chunk = current_audio_chunk
                else:
                    # For subsequent chunks, apply cross-fading or simply concatenate.
                    fade_len = int(self.sr * (params.crossfade_duration_milliseconds / 1000.0))
                    can_fade = (
                        params.crossfade_duration_milliseconds > 0
                        and fade_len > 0
                        and previous_audio_chunk.shape[0] >= fade_len
                        and current_audio_chunk.shape[0] >= fade_len
                    )

                    if can_fade:
                        # --- Cross-fade logic ---
                        # The part of the previous chunk that will be sent before the fade
                        prev_main_body = previous_audio_chunk[:-fade_len]
                        # The tail of the previous chunk that will be faded out
                        prev_tail = previous_audio_chunk[-fade_len:]
                        # The head of the current chunk that will be faded in
                        current_head = current_audio_chunk[:fade_len]
                        # The main body of the current chunk that comes after the fade
                        current_main_body = current_audio_chunk[fade_len:]

                        fade_out = torch.linspace(1.0, 0.0, fade_len, device=self.device)
                        fade_in = torch.linspace(0.0, 1.0, fade_len, device=self.device)
                        mixed_region = (prev_tail * fade_out) + (current_head * fade_in)

                        # The audio to send is the mixed region + the main body of the current chunk
                        output_to_send = torch.cat((mixed_region, current_main_body))
                        # The previous_audio_chunk for the next iteration is the full current_audio_chunk
                        previous_audio_chunk = current_audio_chunk
                    else:
                        # --- No fade logic (concatenate) ---
                        # If no fading, simply send the current chunk
                        output_to_send = current_audio_chunk
                        # The previous_audio_chunk for the next iteration is the full current_audio_chunk
                        previous_audio_chunk = current_audio_chunk

                if output_to_send is not None:
                    await audio_chunk_queue.put(self.audio_processor.to_pcm(output_to_send))

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
        """Streams synthesized audio using a producer-consumer pattern."""
        loop = asyncio.get_running_loop()
        audio_prompt_path = self.voice_manager.get_voice_path(voice_id) if voice_id else None
        conds = await self._prepare_and_get_conds(audio_prompt_path, voice_exaggeration_factor, loop)

        yield self.audio_processor.create_wav_header(self.sr)

        text_chunks = split_text_into_chunks(text, text_processing_chunk_size)
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
            next_chunk_event=next_chunk_event
        )

        # Start producer and consumer tasks
        producer_task = loop.create_task(
            self._t3_producer_task(text_chunks, speech_token_queue, params, self.t3_stream)
        )
        consumer_task = loop.create_task(
            self._s3gen_consumer_task(speech_token_queue, audio_chunk_queue, params, self.s3gen_stream)
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
