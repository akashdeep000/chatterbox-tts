# 1. Start from the official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. Set up environment
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 3. Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 5. Copy the application source code
COPY ./src /app

# 6. Expose the port the API server will run on
EXPOSE 8000

# 7. Define the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
