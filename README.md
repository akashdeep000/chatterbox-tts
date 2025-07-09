# Chatterbox TTS

Chatterbox TTS is a high-performance, containerized Text-to-Speech (TTS) service designed for real-time audio generation and streaming. It supports custom voice cloning and is optimized for deployment on GPU-accelerated platforms like RunPod.

## Project Overview

This project provides a flexible and scalable solution for generating speech from text. It exposes both a RESTful API for generating complete audio files and a WebSocket endpoint for low-latency audio streaming. The system is built to be easily deployable and managed as a serverless endpoint.

## Prerequisites

To run this project, you will need:

*   **Docker**: The application is containerized and requires Docker to be installed.
*   **GPU**: A CUDA-enabled GPU is necessary for the TTS model to perform efficiently.

## Configuration

Before running the application, you must configure the following environment variables. You can set them directly in your shell or create a `.env` file in the project root.

*   `API_KEY`: **(Required)** Your secret API key for securing the service.
*   `CORS_ORIGINS`: A comma-separated list of allowed origins (e.g., `"http://localhost:3000,https://your-frontend.com"`). Defaults to `*` (all origins).
*   `LOG_LEVEL`: The logging level (e.g., `INFO`, `DEBUG`). Defaults to `INFO`.
*   `MODEL_DEVICE`: The device for the TTS model (`cpu`, `cuda`). Defaults to `cpu`.

### Example `.env` file:

```
API_KEY="your-super-secret-api-key"
CORS_ORIGINS="http://localhost:3000,https://your-app.com"
LOG_LEVEL="DEBUG"
```

## Setup and Deployment

### 1. Build the Docker Image

From the project's root directory, build the Docker image:

```bash
docker build -t chatterbox-tts:latest .
```

### 2. Run the Docker Container

Run the container, mapping port `8000`, providing the environment variables, and mounting a volume for persistent voice storage:

```bash
docker run -d -p 8000:8000 \
  --gpus all \
  -v $(pwd)/voices:/home/appuser/voices \
  --env-file .env \
  --name chatterbox-tts \
  chatterbox-tts:latest
```

## Adding Cloned Voices

You can add new voices to the TTS engine by providing audio files. The `clone_voice.py` script copies your audio file into the `voices` directory, which is mounted into the container.

To add a new voice, run the following command on your host machine:

```bash
python3 scripts/clone_voice.py /path/to/your/voice.wav
```

The `voice_id` for your new voice will be the filename (e.g., `voice.wav`).

## API Usage

### REST API: `/tts/generate`

This endpoint generates a complete audio file and returns it as a `.wav` file.

**Example with `curl` (default voice):**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{"text": "Hello, world!"}' \
  http://localhost:8000/tts/generate --output output.wav
```

**Example with `curl` (custom voice):**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{"text": "Hello, this is a custom voice.", "voice_id": "your_voice.wav"}' \
  http://localhost:8000/tts/generate --output custom_voice_output.wav
```

**Live Demo Example:**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{"text": "Hello, this is a test of the live demo."}' \
  http://localhost:8000/tts/generate --output demo_output.wav
```

### WebSocket API: `/tts/stream`

This endpoint streams audio chunks in real-time, making it ideal for low-latency applications.

**Python Example:**

```python
import asyncio
import websockets
import json

async def stream_tts():
    uri = "ws://localhost:8000/tts/stream"
    async with websockets.connect(uri) as websocket:
        request = {
            "text": "This is a real-time audio stream.",
            "voice_id": "your_voice.wav"  # Optional
        }
        await websocket.send(json.dumps(request))

        with open("stream_output.wav", "wb") as f:
            try:
                while True:
                    chunk = await websocket.recv()
                    f.write(chunk)
            except websockets.exceptions.ConnectionClosed:
                print("Stream finished.")

if __name__ == "__main__":
    asyncio.run(stream_tts())
```

### Voice Management API

The service provides a set of RESTful endpoints to manage custom voices.

#### Upload a Voice

*   **Endpoint:** `POST /voices`
*   **Description:** Upload a new voice file. The `voice_id` will be the filename.
*   **Request:** `multipart/form-data` with a file named `voice.wav`.
*   **Headers:** `X-API-Key: <YOUR_API_KEY>`

**Example with `curl`:**

```bash
curl -X POST \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -F "file=@/path/to/your/voice.wav" \
  http://localhost:8000/voices
```

**Success Response (`201 Created`):**

```json
{
  "voice_id": "voice.wav",
  "message": "Voice uploaded successfully."
}
```

#### List Voices

*   **Endpoint:** `GET /voices`
*   **Description:** Get a list of all available voice IDs.
*   **Headers:** `X-API-Key: <YOUR_API_KEY>`

**Example with `curl`:**

```bash
curl -X GET \
  -H "X-API-Key: <YOUR_API_KEY>" \
  http://localhost:8000/voices
```

**Success Response (`200 OK`):**

```json
[
  "voice1.wav",
  "voice2.mp3"
]
```

#### Delete a Voice

*   **Endpoint:** `DELETE /voices/{voice_id}`
*   **Description:** Delete a specific voice by its ID.
*   **Headers:** `X-API-Key: <YOUR_API_KEY>`

**Example with `curl`:**

```bash
curl -X DELETE \
  -H "X-API-Key: <YOUR_API_KEY>" \
  http://localhost:8000/voices/voice.wav
```

**Success Response (`200 OK`):**

```json
{
  "message": "Voice 'voice.wav' deleted successfully."
}
```

## RunPod Deployment Notes

This project is optimized for deployment on cloud platforms like [RunPod](https://runpod.io), where you can easily deploy the container as a serverless GPU endpoint. When deploying on RunPod, ensure you:

1.  Use the Docker image you built (`chatterbox-tts:latest`).
2.  Configure a persistent volume and map it to `/app/voices` in the container.
3.  Expose the container's port `8000`.

This setup allows you to manage a scalable, high-performance TTS service with ease.