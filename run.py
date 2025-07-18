import subprocess
import sys
import os
import torch
import signal
from src.config import settings

def run_worker(gpu_id, port):
    """Launch a uvicorn worker pinned to a specific GPU."""
    env = os.environ.copy()
    worker_id = f"gpu{gpu_id}" if gpu_id is not None else "cpu"
    env["WORKER_ID"] = worker_id
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Starting worker {worker_id} on port {port}")
    else:
        print(f"Starting worker {worker_id} on port {port}")

    command = [
        sys.executable, "-m", "uvicorn",
        "src.main:create_app",
        "--factory",
        "--host", settings.HOST,
        "--port", str(port),
        "--workers", "1", # Each process is a single worker
        "--log-level", settings.LOG_LEVEL.lower()
    ]
    if settings.DEBUG:
        command.append("--reload")

    return subprocess.Popen(command, env=env)

if __name__ == "__main__":
    processes = []
    base_port = settings.PORT

    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Found {gpu_count} GPUs.")
            for i in range(gpu_count):
                # Each worker will listen on a different port
                port = base_port + i
                proc = run_worker(i, port)
                processes.append(proc)
        else:
            print("No GPUs found. Starting a single worker on CPU.")
            proc = run_worker(None, base_port)
            processes.append(proc)

        # Gracefully handle shutdown
        def shutdown_handler(signum, frame):
            for p in processes:
                p.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        # Wait for all processes to complete
        for proc in processes:
            proc.wait()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Shutting down all worker processes.")
        for p in processes:
            p.terminate()
