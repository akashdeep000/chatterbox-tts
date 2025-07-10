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

# Copy source code
COPY ./src ./src
COPY ./static ./static

# Stage 2: Production Environment
# Use a smaller -runtime image which only includes the CUDA runtime
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# Copy the virtual environment and source code from the builder stage
COPY --from=builder /app/venv ./venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/static ./static

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/appuser/venv/bin:$PATH"

# Expose the application port
EXPOSE 8000

# Health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Define the command to run the application
# The working directory is now /home/appuser, so the path to the app is src.main:app
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
