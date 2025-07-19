import uvicorn
import subprocess
import sys
from src.config import settings
import os
from pathlib import Path

if __name__ == "__main__":
    uvicorn.run(
        "src.main:create_app",
        factory=True,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,
        log_level=settings.LOG_LEVEL.lower()
    )
