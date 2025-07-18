import uvicorn
import subprocess
import sys
from src.config import settings
import os
from pathlib import Path

def cleanup_worker_counter():
    """Deletes the worker counter and lock files to ensure a clean start."""
    counter_file = Path("/tmp/worker_counter.txt")
    lock_file = Path("/tmp/worker_counter.lock")
    try:
        if counter_file.exists():
            counter_file.unlink()
            print("Cleaned up worker counter file.")
        if lock_file.exists():
            lock_file.unlink()
            print("Cleaned up worker lock file.")
    except OSError as e:
        print(f"Error cleaning up worker counter files: {e}")

def get_gpu_count():
    """
    Gets the GPU count in a separate process to avoid initializing CUDA in the main process.
    """
    command = [
        sys.executable,
        "-c",
        "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Could not determine GPU count. Defaulting to 1. Error: {e}")
        return 1

if __name__ == "__main__":
    cleanup_worker_counter()
    gpu_count = get_gpu_count()
    if gpu_count > 0:
        workers_count = gpu_count
        print(f"Auto-detected {workers_count} GPUs. Starting {workers_count} workers.")
    else:
        workers_count = 2
        print(f"No GPUs detected. Starting {workers_count} workers.")

    # In debug mode, Uvicorn requires workers=1 to use the --reload flag.
    if settings.DEBUG:
        workers_count = 1
        print("Debug mode enabled. Starting 1 worker with reload.")

    uvicorn.run(
        "src.main:create_app",
        factory=True,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=workers_count,
        log_level=settings.LOG_LEVEL.lower()
    )
