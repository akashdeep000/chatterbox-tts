import asyncio
import uuid
import time
import pickle
import zmq.asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.logging_config import log
import logging

from src.voice_manager import VoiceManager
from src import master, api


# --- Logging Filter ---
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out logs for specific endpoints from uvicorn.access
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
    This now acts as the master process in our multi-process architecture.
    """
    app = FastAPI(debug=settings.DEBUG)
    context = zmq.asyncio.Context()

    @app.on_event("startup")
    async def startup_event():
        """
        Initializes the master process:
        1. Sets up ZeroMQ sockets for IPC.
        2. Spawns worker processes.
        3. Starts the result listener task.
        """
        log.info("Master process starting up...")

        # 1. Setup master-side ZeroMQ sockets
        job_socket, result_socket, broadcast_socket = master.setup_master_sockets(context)

        # Make sockets and other managers available to the API layer
        api.job_socket = job_socket
        api.broadcast_socket = broadcast_socket
        api.voice_manager = VoiceManager() # Master process manages voice files

        # 2. Spawn worker processes and trigger voice cache warming
        master.spawn_workers(broadcast_socket)

        # 3. Start the background task to listen for results from workers
        asyncio.create_task(master.result_listener(result_socket))

        # 4. Initialize NVML for GPU monitoring
        try:
            import pynvml
            pynvml.nvmlInit()
            app.state.pynvml_initialized = True
            log.info("pynvml initialized successfully for system monitoring.")
        except Exception as e:
            app.state.pynvml_initialized = False
            log.warning(f"Could not initialize pynvml. GPU status will not be available. Error: {e}")

        log.info("Master process startup complete.")

    @app.on_event("shutdown")
    def shutdown_event():
        """
        Cleans up resources by terminating worker processes and shutting down NVML.
        """
        log.info("Master process shutting down...")
        master.shutdown_workers()
        context.term()

        if getattr(app.state, 'pynvml_initialized', False):
            try:
                import pynvml
                pynvml.nvmlShutdown()
                log.info("pynvml shut down successfully.")
            except Exception as e:
                log.error(f"Error shutting down pynvml: {e}")

        log.info("Master process shutdown complete.")

    # --- Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # Skip our custom logging for health and system status endpoints
        if request.url.path in ["/health", "/system-status"]:
            return await call_next(request)

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time
        log.info(f"[{request_id}] Handled request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.4f}s")
        response.headers["X-Request-ID"] = request_id
        return response

    # --- Register API Routes ---
    api.register_api_routes(app)

    # --- Static Files ---
    app.mount("/static", StaticFiles(directory="static"), name="static")

    return app

app = create_app()
