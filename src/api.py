"""
API Endpoints for the TTS Server.

This module defines the FastAPI routes for handling TTS generation,
voice management, and system status checks.
"""
from fastapi import FastAPI, HTTPException, Depends, status, Request, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
import asyncio
import pickle
from typing import Optional
import zmq.asyncio
import time

from src.audio_encoding import AudioEncoder
from src.config import settings, tts_config
from src.voice_manager import VoiceManager
from src.ipc import TTSRequest, TTSStreamChunk, BroadcastCommand
from src.master import active_requests
from src.logging_config import log

# These will be initialized in the main app factory
job_socket: zmq.asyncio.Socket
broadcast_socket: zmq.asyncio.Socket
voice_manager: VoiceManager

# --- Security ---
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query_scheme = APIKeyQuery(name="api_key", auto_error=False)

async def get_api_key(
    api_key_header: Optional[str] = Depends(api_key_header_scheme),
    api_key_query: Optional[str] = Depends(api_key_query_scheme),
):
    api_key = api_key_header or api_key_query
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key

class TTSRequestModel(BaseModel):
    text: str
    voice_id: Optional[str] = None
    format: Optional[str] = "wav"
    cfg_guidance_weight: float = tts_config.CFG_GUIDANCE_WEIGHT
    synthesis_temperature: float = tts_config.SYNTHESIS_TEMPERATURE
    text_processing_chunk_size: int = tts_config.TEXT_PROCESSING_CHUNK_SIZE
    audio_tokens_per_slice: int = tts_config.AUDIO_TOKENS_PER_SLICE
    remove_trailing_milliseconds: int = tts_config.REMOVE_TRAILING_MILLISECONDS
    remove_leading_milliseconds: int = tts_config.REMOVE_LEADING_MILLISECONDS
    chunk_overlap_strategy: str = tts_config.CHUNK_OVERLAP_STRATEGY
    crossfade_duration_milliseconds: int = tts_config.CROSSFADE_DURATION_MILLISECONDS

def register_api_routes(app: FastAPI):
    """Registers all the API routes with the FastAPI application."""

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def read_root():
        return "static/index.html"

    @app.api_route("/tts/generate", methods=["GET", "POST"], dependencies=[Depends(get_api_key)])
    async def tts_generate(request: Request):
        if request.method == "POST":
            try:
                body = await request.json()
                tts_request = TTSRequestModel(**body)
            except Exception:
                return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)
        else: # GET
            tts_request = TTSRequestModel(**request.query_params)

        if not tts_request.text:
            return JSONResponse(content={"error": "Text is required"}, status_code=400)

        request_id = request.state.request_id
        # Create a queue with a max size to apply backpressure
        result_queue = asyncio.Queue(maxsize=2000)
        active_requests[request_id] = result_queue

        ipc_request = TTSRequest(
            request_id=request_id,
            text=tts_request.text,
            output_format=tts_request.format,
            voice_id=tts_request.voice_id,
            cfg_guidance_weight=tts_request.cfg_guidance_weight,
            synthesis_temperature=tts_request.synthesis_temperature,
            text_processing_chunk_size=tts_request.text_processing_chunk_size,
            audio_tokens_per_slice=tts_request.audio_tokens_per_slice,
            remove_trailing_milliseconds=tts_request.remove_trailing_milliseconds,
            remove_leading_milliseconds=tts_request.remove_leading_milliseconds,
            chunk_overlap_strategy=tts_request.chunk_overlap_strategy,
            crossfade_duration_milliseconds=tts_request.crossfade_duration_milliseconds
        )

        await job_socket.send(pickle.dumps(ipc_request))

        async def stream_generator():
            try:
                while True:
                    result: TTSStreamChunk = await result_queue.get()
                    if result.is_final:
                        break

                    # This yield will raise an exception if the client disconnects.
                    yield result.chunk

                    # Explicitly yield control to the event loop after every chunk.
                    # This prevents this loop from starving the event loop if the
                    # client is consuming very quickly.
                    await asyncio.sleep(0)
            finally:
                # This block runs when the client disconnects OR the stream finishes.
                # We broadcast the cancellation first to stop the worker from sending more data.
                log.info(f"Broadcasting cancellation/cleanup command for request_id: {request_id}")
                command = BroadcastCommand(command="cancel_request", details={"request_id": request_id})
                await broadcast_socket.send(pickle.dumps(command))

                # Give the command a moment to propagate before we clean up the master's state.
                await asyncio.sleep(0.1)

                # Now, clean up the request from the master's perspective.
                if request_id in active_requests:
                    del active_requests[request_id]

        try:
            # Instantiate a temporary encoder just to get the correct MIME type.
            # The sample_rate is required for the constructor but not for get_mime_type.
            temp_encoder = AudioEncoder(output_format=tts_request.format, sample_rate=24000)
            media_type = temp_encoder.get_mime_type()
        except ValueError:
            # This will be raised by AudioFormat(output_format.lower()) if the format is invalid
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audio format: '{tts_request.format}'. Supported formats are: wav, raw_pcm, fmp4, mp3, webm"
            )

        return StreamingResponse(stream_generator(), media_type=media_type)

    @app.post("/voices", dependencies=[Depends(get_api_key)])
    async def upload_voice(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            voice_manager.save_voice(file.filename, contents)
            # Broadcast a command to all workers to warm up the cache for the new voice
            log.info(f"Broadcasting warm-up command for new voice: {file.filename}")
            command = BroadcastCommand(command="warm_up_voices", details={"voice_ids": [file.filename]})
            await broadcast_socket.send(pickle.dumps(command))
            return JSONResponse(content={"voice_id": file.filename, "message": "Voice uploaded and cache warming initiated."}, status_code=201)
        except FileExistsError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            return JSONResponse(content={"error": "Internal server error"}, status_code=500)

    @app.get("/voices", dependencies=[Depends(get_api_key)])
    async def list_voices():
        return voice_manager.list_voices()

    @app.delete("/voices/{voice_id}", dependencies=[Depends(get_api_key)])
    async def delete_voice(voice_id: str):
        try:
            voice_manager.delete_voice(voice_id)
            command = BroadcastCommand(command="clear_voice_cache", details={"voice_id": voice_id})
            await broadcast_socket.send(pickle.dumps(command))
            return JSONResponse(content={"message": f"Voice '{voice_id}' deleted successfully."})
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found.")
        except Exception as e:
            return JSONResponse(content={"error": "Internal server error"}, status_code=500)

    @app.get("/health")
    async def health_check():
        # In a multi-process architecture, the master's health indicates the API is responsive.
        # A more advanced health check could query workers via the IPC channel.
        return {"status": "ok", "message": "Master process is running."}

    @app.get("/system-status", dependencies=[Depends(get_api_key)])
    async def system_status(request: Request):
        """
        Provides real-time CPU, RAM, and GPU utilization details.
        """
        # --- CPU and RAM Info ---
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_info = psutil.virtual_memory()
            ram_total_gb = ram_info.total / (1024**3)
            ram_used_gb = ram_info.used / (1024**3)
            ram_free_gb = ram_info.free / (1024**3)

            cpu_data = {
                "utilization_percent": cpu_percent,
                "ram_gb": {
                    "total": round(ram_total_gb, 2),
                    "used": round(ram_used_gb, 2),
                    "free": round(ram_free_gb, 2),
                    "percent_used": ram_info.percent
                }
            }
        except ImportError:
            cpu_data = {"error": "psutil library not installed."}
        except Exception as e:
            cpu_data = {"error": f"Could not retrieve CPU/RAM stats: {e}"}

        # --- GPU Info ---
        gpus_data = []
        if not getattr(request.app.state, 'pynvml_initialized', False):
            gpus_data = []
        else:
            try:
                import pynvml
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info = {
                        "device_id": i,
                        "utilization_percent": {
                            "gpu": utilization.gpu,
                            "memory": utilization.memory
                        },
                        "memory_gb": {
                            "total": round(mem_info.total / (1024**3), 2),
                            "used": round(mem_info.used / (1024**3), 2),
                            "free": round(mem_info.free / (1024**3), 2)
                        }
                    }
                    gpus_data.append(gpu_info)
            except Exception as e:
                gpus_data = []

        return {"cpu": cpu_data, "gpus": gpus_data}