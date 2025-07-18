import uvicorn
import torch
from src.config import settings
import os

if __name__ == "__main__":
    workers_count = 1
    if torch.cuda.is_available():
        workers_count = torch.cuda.device_count()
        print(f"Found {workers_count} GPUs. Starting {workers_count} workers.")
    else:
        print("No GPUs found. Starting 1 worker on CPU.")

    # In debug mode, Uvicorn requires workers=1 to use the --reload flag.
    if settings.DEBUG:
        workers_count = 1
        print("Debug mode enabled. Starting 1 worker with reload.")

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=workers_count,
        log_level=settings.LOG_LEVEL.lower()
    )
