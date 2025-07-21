import sys
from loguru import logger
from .config import settings

def configure_logging(worker_id: str = "M", device: str = "CPU"):
    """
    Set up Loguru logger using a custom formatter function for robustness.
    This function should be called once per process.
    """
    logger.remove()

    # Standardize device name for logging, e.g., "cuda:0" -> "GPU-0"
    device_name = f"GPU-{device.split(':')[-1]}" if "cuda" in device else "CPU"

    def formatter(record):
        """
        This closure captures the worker_id and device_name and injects
        them into the record's 'extra' dict before formatting.
        """
        record["extra"]["device"] = device_name
        record["extra"]["worker_id"] = worker_id

        # The format string is now part of the function
        return (
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<cyan>[{extra[device]: <5}|W-{extra[worker_id]: <2}]</cyan> | "
            "<level>{level: <7}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>\n"
        )

    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=formatter,
        colorize=True,
    )

# The logger instance is imported directly by other modules.
# They will then call configure_logging() to set it up.
log = logger