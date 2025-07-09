import uvicorn
from src.config import settings

if __name__ == "__main__":
    """
    This script provides a convenient way to run the application for local development.

    It launches the Uvicorn server with the correct application path and enables --reload,
    which automatically restarts the server when code changes are detected.

    To run the application, execute this command from the project root:
    python run.py
    """
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )