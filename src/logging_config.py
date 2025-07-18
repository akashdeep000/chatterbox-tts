import sys
from loguru import logger
from .config import settings

def setup_logging():
    """
    Set up Loguru logger with a specific format and colorization.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        # format=(
        #     "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        #     "<level>{level: <7}</level> | "
        #     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        # ),
        format=(
            "<level>{level: <7}</level> | <level>{message}</level>"
        ),
        colorize=True,
    )
    return logger

log = setup_logging()