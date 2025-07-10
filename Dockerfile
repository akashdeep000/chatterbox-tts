# Stage 1: Production Environment
# Use a single-stage build with the runtime image
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies, Python, and build tools.
# The runtime image may not have all the tools needed to build python packages,
# so we install build-essential.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    libsndfile1 \
    libgomp1 \
    curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY ./src ./src
COPY ./static ./static
COPY ./scripts ./scripts
COPY run.py .

RUN python3 scripts/download_models.py

# Create a non-root user and change ownership of the app directory
RUN useradd --create-home appuser && chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Set environment variables
# PYTHONUNBUFFERED is already set at the top of the file.
# PATH is already set to use the venv.
ENV MODEL_PATH="/app/models"

# Expose the application port
EXPOSE 8000

# Health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Define the command to run the application
CMD ["python3", "run.py"]
