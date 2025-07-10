# Stage 1: Builder Environment
# Use a -devel image which includes the full CUDA toolkit and compilers
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    libsndfile1 && \
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

# Download models
RUN python3 scripts/download_models.py

# Stage 2: Production Environment
# Use a smaller -runtime image which only includes the CUDA runtime
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Install python3 in the production image
USER root
RUN apt-get update && apt-get install -y python3 libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# Copy the virtual environment, source code, and models from the builder stage
COPY --from=builder /app/venv ./venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/static ./static
COPY --from=builder /app/models ./models
COPY --from=builder /app/run.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/appuser/venv/bin:$PATH"
ENV MODEL_PATH="/home/appuser/models"

# Expose the application port
EXPOSE 8000

# Health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Define the command to run the application
CMD ["python3", "run.py"]
