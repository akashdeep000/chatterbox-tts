import warnings
# Suppress the specific UserWarning from pkg_resources
# This is coming from a dependency (perth) and is safe to ignore for now.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# Suppress the specific UserWarning from pkg_resources
# This is coming from a dependency (perth) and is safe to ignore for now.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import logging
import time
from typing import Optional
import asyncio
import uuid
from concurrent.futures import ProcessPoolExecutor

from .config import settings, tts_config
import torch
from .dependencies import get_tts_engine_manager, get_voice_manager
from . import dependencies
from .tts_streaming import TTSEngineManager, InitializationState
from .voice_manager import VoiceManager
from fastapi import File, UploadFile

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out logs for specific endpoints
        if hasattr(record, 'args') and isinstance(record.args, tuple) and len(record.args) > 2:
            path = record.args[2]
            if isinstance(path, str) and (path.startswith("/health") or path.startswith("/system-status")):
                return False
        return True

# Add the filter to Uvicorn's access logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

def create_app() -> FastAPI:
    """
    Application factory to create and configure the FastAPI app.
    This is used by Uvicorn to ensure resources are initialized within each worker process.
    """
    app = FastAPI(debug=settings.DEBUG)

    @app.on_event("startup")
    async def startup_event():
        """
        Initializes the TTS Engine Manager, Voice Manager, and NVML for GPU monitoring.
        """
        # Create a shared process pool for CPU-bound work.
        # The number of workers is configured via the CPU_WORKER_COUNT setting.
        app.state.process_pool = ProcessPoolExecutor(max_workers=settings.CPU_WORKER_COUNT)
        logger.info(f"Initialized ProcessPoolExecutor with {settings.CPU_WORKER_COUNT} workers.")

        # --- NVML Initialization ---
        try:
            import pynvml
            pynvml.nvmlInit()
            app.state.pynvml_initialized = True
            logger.info("pynvml initialized successfully.")
        except Exception as e:
            app.state.pynvml_initialized = False
            logger.warning(f"Could not initialize pynvml. GPU status will not be available. Error: {e}")

        # --- TTS Engine Manager and Voice Manager Initialization ---
        logger.info("Initializing TTS Engine Manager and Voice Manager...")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        dependencies.tts_engine_manager = TTSEngineManager(num_gpus=num_gpus)
        dependencies.voice_manager = VoiceManager()
        dependencies.segmenter = pysbd.Segmenter(language="en", clean=False)
        await dependencies.tts_engine_manager.ainit()
        logger.info("TTS Engine Manager, Voice Manager, and Segmenter initialized successfully.")

        # --- Voice Cache Warm-up ---
        logger.info("Warming up voice cache for all engines...")
        available_voices = dependencies.voice_manager.list_voices()
        if available_voices:
            for voice_id in available_voices:
                try:
                    voice_path = dependencies.voice_manager.get_voice_path(voice_id)
                    if voice_path:
                        # Warm up cache on all engines
                        for engine in dependencies.tts_engine_manager.get_all_engines():
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(None, engine.prepare_conditionals, voice_path)
                            log_prefix = f"GPU {engine.gpu_id}" if engine.device != "cpu" else "CPU"
                            logger.info(f"Preloaded voice '{voice_id}' into cache for {log_prefix}.")
                    else:
                        logger.warning(f"Could not find path for voice ID '{voice_id}'. Skipping.")
                except Exception as e:
                    logger.error(f"Failed to preload voice '{voice_id}': {e}", exc_info=True)
            logger.info("Voice cache warm-up complete.")
        else:
            logger.info("No voices found to preload into cache.")

    @app.on_event("shutdown")
    def shutdown_event():
        """
        Cleans up resources, like shutting down NVML.
        """
        # Clean up the process pool
        if hasattr(app.state, 'process_pool'):
            app.state.process_pool.shutdown(wait=True)

        if getattr(app.state, 'pynvml_initialized', False):
            try:
                import pynvml
                pynvml.nvmlShutdown()
                logger.info("pynvml shut down successfully.")
            except Exception as e:
                logger.error(f"Error shutting down pynvml: {e}")


    # --- Static Files ---
    app.mount("/static", StaticFiles(directory="static"), name="static")


    # --- Middleware ---

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Logging Middleware
    # --- Security ---
    api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
    api_key_query_scheme = APIKeyQuery(name="api_key", auto_error=False)

    async def get_api_key(
        api_key_header: Optional[str] = Depends(api_key_header_scheme),
        api_key_query: Optional[str] = Depends(api_key_query_scheme),
    ):
        """
        Get API key from header or query string.
        """
        api_key = api_key_header or api_key_query

        if not api_key or api_key != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key"
            )
        return api_key

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # Skip logging for health and system status endpoints
        if request.url.path in ["/health", "/system-status"]:
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(
            f'[{request_id}] Handled request: {request.method} {request.url.path} - '
            f'Status: {response.status_code} - Duration: {duration:.4f}s'
        )

        response.headers["X-Request-ID"] = request_id
        if request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


    @app.get("/", response_class=FileResponse)
    async def read_root():
        return "static/index.html"


    @app.get("/health")
    async def health_check():
        """
        Health check endpoint to verify the application is running and TTS engine status.
        """
        if dependencies.tts_engine_manager:
            tts_status = dependencies.tts_engine_manager.get_status()
            # Check if any engine has an error
            for gpu_id, status_info in tts_status.items():
                if status_info["state"] == InitializationState.ERROR.value:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"TTS Engine Error on {gpu_id}: {status_info['error']}")
            # Check if all engines are ready
            all_ready = all(s["state"] == InitializationState.READY.value for s in tts_status.values())
            if not all_ready:
                 return JSONResponse(content={"status": "initializing", "tts_engines": tts_status}, status_code=status.HTTP_202_ACCEPTED)
        else:
            tts_status = {"state": InitializationState.NOT_STARTED.value, "progress": "TTS engine manager not yet instantiated", "error": None}
            return JSONResponse(content={"status": "not_started", "tts_engines": tts_status}, status_code=status.HTTP_202_ACCEPTED)

        return {"status": "ok", "tts_engines": tts_status}

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
        gpus_data = []
        if not getattr(app.state, 'pynvml_initialized', False):
            gpus_data = []
        else:
            try:
                import pynvml
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count == 0:
                    gpus_data = []
                else:
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
            except pynvml.NVMLError as e:
                logger.error(f"NVIDIA driver/library error during status check: {e}", exc_info=True)
                gpus_data = []
            except ImportError:
                gpus_data = []
            except Exception as e:
                logger.error(f"An unexpected error occurred while fetching GPU status: {e}", exc_info=True)
                gpus_data = []


        return {"cpu": cpu_data, "gpus": gpus_data}



    class TTSRequest(BaseModel):
        text: str
        voice_id: Optional[str] = None
        format: Optional[str] = None
        # voice_exaggeration_factor: float = tts_config.VOICE_EXAGGERATION_FACTOR # can't use this for efficient caching
        cfg_guidance_weight: float = tts_config.CFG_GUIDANCE_WEIGHT
        synthesis_temperature: float = tts_config.SYNTHESIS_TEMPERATURE
        text_processing_chunk_size: Optional[int] = tts_config.TEXT_PROCESSING_CHUNK_SIZE
        audio_tokens_per_slice: Optional[int] = tts_config.AUDIO_TOKENS_PER_SLICE
        remove_trailing_milliseconds: int = tts_config.REMOVE_TRAILING_MILLISECONDS
        remove_leading_milliseconds: int = tts_config.REMOVE_LEADING_MILLISECONDS
        chunk_overlap_strategy: str = tts_config.CHUNK_OVERLAP_STRATEGY
        crossfade_duration_milliseconds: int = tts_config.CROSSFADE_DURATION_MILLISECONDS

    def get_output_format_and_media_type(
        accept_header: Optional[str],
        request_format: Optional[str] = None
    ) -> (str, str):
        """
        Determines the output format and media type, prioritizing the request_format
        parameter over the Accept header.
        """
        format_map = {
            "mp3": "audio/mpeg",
            "fmp4": "audio/mp4",
            "raw_pcm": "audio/pcm",
            "webm": "audio/webm",
            "wav": "audio/wav"
        }

        # Prioritize the format from the request parameter
        if request_format and request_format in format_map:
            return request_format, format_map[request_format]

        # Fallback to Accept header
        if accept_header:
            if "audio/mpeg" in accept_header:
                return "mp3", "audio/mpeg"
            if "video/mp4" in accept_header or "audio/mp4" in accept_header:
                return "fmp4", "audio/mp4"
            if "audio/pcm" in accept_header:
                return "raw_pcm", "audio/pcm"
            if "audio/webm" in accept_header:
                return "webm", "audio/webm"

        # Default to wav
        return "wav", "audio/wav"


    @app.api_route("/tts/generate", methods=["GET", "POST"], dependencies=[Depends(get_api_key)])
    async def tts_generate(
        request: Request,
        tts_engine_manager: TTSEngineManager = Depends(get_tts_engine_manager)
    ):
        """
        Generates and streams audio for the given text.
        Accepts both GET and POST requests.
        """
        tts_engine = await tts_engine_manager.get_engine()
        if request.method == "POST":
            try:
                body = await request.json()
                tts_request = TTSRequest(**body)
            except Exception:
                return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)
        else:  # GET request
            query_params = request.query_params
            def get_param(name, default, cast_func):
                val = query_params.get(name)
                if val is None or val == '':
                    return default
                try:
                    return cast_func(val)
                except (ValueError, TypeError):
                    return default

            tts_request = TTSRequest(
                text=query_params.get("text"),
                voice_id=query_params.get("voice_id"),
                format=query_params.get("format"),
                cfg_guidance_weight=get_param("cfg_guidance_weight", tts_config.CFG_GUIDANCE_WEIGHT, float),
                synthesis_temperature=get_param("synthesis_temperature", tts_config.SYNTHESIS_TEMPERATURE, float),
                text_processing_chunk_size=get_param("text_processing_chunk_size", tts_config.TEXT_PROCESSING_CHUNK_SIZE, int),
                audio_tokens_per_slice=get_param("audio_tokens_per_slice", tts_config.AUDIO_TOKENS_PER_SLICE, int),
                remove_trailing_milliseconds=get_param("remove_trailing_milliseconds", tts_config.REMOVE_TRAILING_MILLISECONDS, int),
                remove_leading_milliseconds=get_param("remove_leading_milliseconds", tts_config.REMOVE_LEADING_MILLISECONDS, int),
                chunk_overlap_strategy=get_param("chunk_overlap_strategy", tts_config.CHUNK_OVERLAP_STRATEGY, str),
                crossfade_duration_milliseconds=get_param("crossfade_duration_milliseconds", tts_config.CROSSFADE_DURATION_MILLISECONDS, int),
            )

        if not tts_request.text:
            return JSONResponse(content={"error": "Text is required"}, status_code=400)

        accept_header = request.headers.get("Accept")
        output_format, media_type = get_output_format_and_media_type(accept_header, tts_request.format)

        request_id = request.state.request_id
        logger.info(f"[{request_id}] Received TTS request for voice_id: {tts_request.voice_id} (format: {output_format}). Waiting for semaphore...")

        try:
            await tts_engine.tts_semaphore.acquire()
            logger.info(f"[{request_id}] Semaphore acquired. Starting TTS generation.")
            start_time = time.time()

            async def stream_generator():
                try:
                    audio_stream = tts_engine.stream(
                        text=tts_request.text,
                        output_format=output_format,
                        voice_id=tts_request.voice_id,
                        cfg_guidance_weight=tts_request.cfg_guidance_weight,
                        synthesis_temperature=tts_request.synthesis_temperature,
                        text_processing_chunk_size=tts_request.text_processing_chunk_size,
                        audio_tokens_per_slice=tts_request.audio_tokens_per_slice,
                        remove_trailing_milliseconds=tts_request.remove_trailing_milliseconds,
                        remove_leading_milliseconds=tts_request.remove_leading_milliseconds,
                        chunk_overlap_strategy=tts_request.chunk_overlap_strategy,
                        crossfade_duration_milliseconds=tts_request.crossfade_duration_milliseconds,
                        start_time=start_time,
                        request_id=request_id,
                        request=request
                    )
                    async for chunk in audio_stream:
                        yield chunk
                finally:
                    tts_engine.tts_semaphore.release()
                    logger.info(f"[{request_id}] Semaphore released. TTS generation finished.")

            return StreamingResponse(stream_generator(), media_type=media_type)
        except Exception as e:
            logger.error(f"[{request_id}] TTS generation failed: {e}", exc_info=True)
            # Ensure semaphore is released in case of an error before streaming starts
            if tts_engine.tts_semaphore.locked():
                tts_engine.tts_semaphore.release()
            return JSONResponse(content={"error": "Internal server error"}, status_code=500)


    @app.post("/voices", dependencies=[Depends(get_api_key)])
    async def upload_voice(
        file: UploadFile = File(...),
        voice_manager: VoiceManager = Depends(get_voice_manager),
        tts_engine_manager: TTSEngineManager = Depends(get_tts_engine_manager)
    ):
        """
        Upload a new voice file.
        """
        try:
            contents = await file.read()
            voice_manager.save_voice(file.filename, contents)
            # Invalidate cache on all engines
            for engine in tts_engine_manager.get_all_engines():
                engine.clear_voice_cache(file.filename)
            return JSONResponse(content={"voice_id": file.filename, "message": "Voice uploaded successfully."}, status_code=201)
        except FileExistsError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            logger.error(f"Voice upload failed: {e}", exc_info=True)
            return JSONResponse(content={"error": "Internal server error"}, status_code=500)

    @app.get("/voices", dependencies=[Depends(get_api_key)])
    async def list_voices(voice_manager: VoiceManager = Depends(get_voice_manager)):
        """
        Get a list of all available voices.
        """
        return voice_manager.list_voices()

    @app.delete("/voices/{voice_id}", dependencies=[Depends(get_api_key)])
    async def delete_voice(
        voice_id: str,
        voice_manager: VoiceManager = Depends(get_voice_manager),
        tts_engine_manager: TTSEngineManager = Depends(get_tts_engine_manager)
    ):
        """
        Delete a specific voice by its ID.
        """
        try:
            voice_manager.delete_voice(voice_id)
            # Invalidate cache on all engines
            for engine in tts_engine_manager.get_all_engines():
                engine.clear_voice_cache(voice_id)
            return JSONResponse(content={"message": f"Voice '{voice_id}' deleted successfully."})
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found.")
        except Exception as e:
            logger.error(f"Voice deletion failed: {e}", exc_info=True)
            return JSONResponse(content={"error": "Internal server error"}, status_code=500)

    return app

# This allows running the app directly with `uvicorn src.main:app` for basic testing,
# while the factory pattern is used for multi-process production runs.
app = create_app()
