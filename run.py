import uvicorn
from src.config import settings
import os

if __name__ == "__main__":
    # Get worker count from settings. Default is 1.
    workers_count = settings.WORKERS_COUNT

    # In debug mode, Uvicorn requires workers=1 to use the --reload flag.
    if settings.DEBUG:
        workers_count = 1

    uvicorn.run(
        "src.main:create_app",  # Point to the factory function
        factory=True,           # Tell Uvicorn to use the factory
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=workers_count,
        log_level=settings.LOG_LEVEL.lower()
    )
