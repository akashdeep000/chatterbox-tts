import uvicorn
import subprocess
import sys
from src.config import settings
from src.logging_config import configure_logging
import os
from pathlib import Path

if __name__ == "__main__":
    # Configure logging for the main Uvicorn/FastAPI process.
    # This MUST be done before Uvicorn loads the application.
    configure_logging(worker_id="M", device="CPU")

    uvicorn.run(
        "src.main:create_app",
        factory=True,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,
        log_level=settings.LOG_LEVEL.lower()
    )
