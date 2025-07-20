"""
TTS Worker Process

This module defines the entry point for a single TTS worker process. Each worker
initializes its own TextToSpeechEngine, binds it to a specific GPU device, and
will eventually listen for jobs from the master process via an IPC channel.
"""
import asyncio
import torch
import zmq.asyncio
import pickle
from src.tts_streaming import TextToSpeechEngine
from src.logging_config import configure_logging, log
from src.ipc import setup_worker_sockets, TTSRequest, TTSStreamChunk
from src.tts_streaming import CancellationToken
from typing import Dict

# Worker-specific mapping of request_id to its cancellation token
active_cancellations: Dict[str, CancellationToken] = {}


async def handle_request(engine: TextToSpeechEngine, result_socket: zmq.asyncio.Socket, request: TTSRequest):
    """Handles a single TTS request and streams the results back."""
    log.info(f"Processing request {request.request_id}...")
    loop = asyncio.get_running_loop()
    cancellation_token = CancellationToken(loop)
    active_cancellations[request.request_id] = cancellation_token

    try:
        audio_generator = engine.stream(
            text=request.text,
            output_format=request.output_format,
            voice_id=request.voice_id,
            cfg_guidance_weight=request.cfg_guidance_weight,
            synthesis_temperature=request.synthesis_temperature,
            text_processing_chunk_size=request.text_processing_chunk_size,
            audio_tokens_per_slice=request.audio_tokens_per_slice,
            remove_trailing_milliseconds=request.remove_trailing_milliseconds,
            remove_leading_milliseconds=request.remove_leading_milliseconds,
            chunk_overlap_strategy=request.chunk_overlap_strategy,
            crossfade_duration_milliseconds=request.crossfade_duration_milliseconds,
            request_id=request.request_id,
            cancellation_token=cancellation_token
        )

        async for chunk in audio_generator:
            result = TTSStreamChunk(request_id=request.request_id, chunk=chunk)
            await result_socket.send(pickle.dumps(result))

        # Send a final message to indicate the end of the stream
        final_result = TTSStreamChunk(request_id=request.request_id, chunk=b"", is_final=True)
        await result_socket.send(pickle.dumps(final_result))

    except Exception as e:
        log.error(f"Error processing request {request.request_id}: {e}", exc_info=True)
        # Optionally, send an error message back to the master
    finally:
        # Ensure the token is removed from the mapping when the request is done
        if request.request_id in active_cancellations:
            del active_cancellations[request.request_id]
            log.info(f"Cleaned up cancellation token for request {request.request_id}")

async def listen_for_jobs(engine: TextToSpeechEngine, job_socket: zmq.asyncio.Socket, result_socket: zmq.asyncio.Socket):
    """Continuously listens for jobs and creates tasks to handle them."""
    while True:
        job_payload = await job_socket.recv()
        request = pickle.loads(job_payload)
        asyncio.create_task(handle_request(engine, result_socket, request))

async def main(worker_id: int, device: str):
    """
    The main asynchronous function for a worker process.
    """
    log.info(f"Initializing TTS engine on device: {device}...")
    context = zmq.asyncio.Context()

    try:
        # Each worker gets its own independent engine, bound to the assigned device.
        engine = TextToSpeechEngine(device=device)
        await engine.ainit()

        log.info(f"Engine initialized successfully on {device}.")

        job_socket, result_socket, broadcast_socket = setup_worker_sockets(context)
        log.info("Sockets connected. Listening for jobs...")

        # Start the job listener and broadcast listener tasks
        listener_tasks = [
            listen_for_jobs(engine, job_socket, result_socket),
            listen_for_broadcasts(engine, broadcast_socket)
        ]
        await asyncio.gather(*listener_tasks)

    except Exception as e:
        log.error(f"Failed to initialize or run engine on {device}: {e}", exc_info=True)
    finally:
        if 'engine' in locals() and engine:
            engine.shutdown()
        log.info("Shutting down...")
        context.term()

async def listen_for_broadcasts(engine: TextToSpeechEngine, broadcast_socket: zmq.asyncio.Socket):
    """Listens for broadcast commands from the master and updates the engine state."""
    while True:
        try:
            command_payload = await broadcast_socket.recv()
            command = pickle.loads(command_payload)

            if command.command == "clear_voice_cache":
                voice_id = command.details.get("voice_id")
                engine.clear_voice_cache(voice_id)
                log.info(f"Cleared voice cache for '{voice_id}' as per broadcast command.")

            elif command.command == "cancel_request":
                request_id = command.details.get("request_id")
                if request_id in active_cancellations:
                    active_cancellations[request_id].cancel()
                    log.info(f"Cancellation signal sent for request_id: {request_id}")
                else:
                    log.warning(f"Received cancellation for unknown or completed request_id: {request_id}")

            elif command.command == "warm_up_voices":
                voice_ids = command.details.get("voice_ids", [])
                log.info(f"Received broadcast to warm up voice cache for: {voice_ids}")
                loop = asyncio.get_running_loop()
                for voice_id in voice_ids:
                    voice_path = engine.voice_manager.get_voice_path(voice_id)
                    if voice_path:
                        await loop.run_in_executor(engine.voice_conditioning_executor, engine.prepare_conditionals, voice_path)
                        log.info(f"Preloaded voice '{voice_id}' into cache.")
                log.info("Voice cache warm-up complete.")

        except Exception as e:
            log.error(f"Error processing broadcast command: {e}", exc_info=True)

if __name__ == "__main__":
    import sys

    # The master process will pass the worker_id and device as command-line arguments.
    if len(sys.argv) != 3:
        print("Usage: python -m src.worker <worker_id> <device>")
        sys.exit(1)

    worker_id_arg = int(sys.argv[1])
    device_arg = sys.argv[2]

    # Configure logging for this specific worker process. This must be done
    # before any log messages are emitted.
    configure_logging(worker_id=str(worker_id_arg), device=device_arg)

    # Set the CUDA device for this specific process. This is crucial.
    if "cuda" in device_arg:
        torch.cuda.set_device(device_arg)

    asyncio.run(main(worker_id=worker_id_arg, device=device_arg))