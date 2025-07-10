# Chatterbox TTS

Chatterbox TTS is a high-performance, containerized Text-to-Speech (TTS) service designed for real-time audio generation and streaming. It supports custom voice cloning and is optimized for deployment on GPU-accelerated platforms like RunPod.

## Project Overview

This project provides a flexible and scalable solution for generating speech from text. It exposes a RESTful API for generating complete audio files and is built to be easily deployable and managed as a serverless endpoint.

## Prerequisites

To run this project, you will need:

*   **Docker**: The application is containerized and requires Docker to be installed.
*   **GPU**: A CUDA-enabled GPU is necessary for the TTS model to perform efficiently.

## Configuration

Before running the application, you must configure the following environment variables. You can set them directly in your shell or create a `.env` file in the project root.

*   `API_KEY`: **(Required)** Your secret API key for securing the service.
*   `CORS_ORIGINS`: A comma-separated list of allowed origins (e.g., `"http://localhost:3000,https://your-frontend.com"`). Defaults to `*` (all origins).
*   `LOG_LEVEL`: The logging level (e.g., `INFO`, `DEBUG`). Defaults to `INFO`.

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
  -v $(pwd)/voices:/app/voices \
  --env-file .env \
  --name chatterbox-tts \
  chatterbox-tts:latest
```

## Docker Hub and GitHub Actions

### Using the Pre-built Docker Image

Instead of building the Docker image locally, you can use the pre-built image from Docker Hub, which is automatically updated with the latest changes.

**1. Pull the Image**

```bash
docker pull akashdeep000/chatterbox-tts:latest
```

**2. Run the Container**

Use the pulled image to run the container:

```bash
docker run -d -p 8000:8000 \
  --gpus all \
  -v $(pwd)/voices:/app/voices \
  --env-file .env \
  --name chatterbox-tts \
  akashdeep000/chatterbox-tts:latest
```

### Automated Builds with GitHub Actions

This repository uses GitHub Actions to automate the building and publishing of the Docker image to Docker Hub. On every push, a new image is built and tagged with `latest` and the commit SHA.

You can view the workflow configuration at [`.github/workflows/publish-docker.yml`](.github/workflows/publish-docker.yml:1).

#### Setup

To enable the workflow to publish to your Docker Hub account, you need to configure the following repository secrets and variables in your GitHub repository settings:

1.  **`DOCKERHUB_USERNAME`**:
    *   **Type**: Variable
    *   **Value**: Your Docker Hub username.
    *   Go to `Settings` > `Secrets and variables` > `Actions` > `Variables` and add a new repository variable.

2.  **`DOCKERHUB_TOKEN`**:
    *   **Type**: Secret
    *   **Value**: Your Docker Hub access token. You can generate one in your Docker Hub account settings.
    *   Go to `Settings` > `Secrets and variables` > `Actions` > `Secrets` and add a new repository secret.


## API Usage

### TTS Generation: `/tts/generate`

This endpoint generates and streams audio in real-time. It supports both `GET` and `POST` requests, providing flexibility for different use cases.

#### Authentication

For all requests to this endpoint, the API key can be provided in one of two ways:
*   **Header:** `X-API-Key: <YOUR_API_KEY>`
*   **Query Parameter:** `?api_key=<YOUR_API_KEY>`

#### POST Request

This method is suitable for server-to-server communication or when the text is long.

**Example with `curl`:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{"text": "Hello, this is a custom voice.", "voice_id": "your_voice.wav"}' \
  http://localhost:8000/tts/generate --output custom_voice_output.wav
```

#### GET Request

This method is ideal for use in web browsers, as it allows you to set the endpoint URL directly as the `src` of an `<audio>` tag.

**Example with `curl`:**
```bash
curl -X GET "http://localhost:8000/tts/generate?text=Hello%20world&voice_id=your_voice.wav&api_key=<YOUR_API_KEY>" --output output.wav
```

**Example in HTML:**
```html
<audio controls src="http://localhost:8000/tts/generate?text=Hello%20world&api_key=<YOUR_API_KEY>"></audio>
```

### Performance Optimizations

The TTS engine is optimized for real-time performance through several mechanisms:
*   **Model Pre-compilation:** The underlying TTS model is pre-compiled when the service starts, reducing latency on all subsequent requests.
*   **Voice Caching:** When a custom voice is used for the first time, its audio characteristics are processed and cached. Subsequent requests with the same voice will use the cached data, resulting in significantly faster audio generation.
*   **Cache Invalidation:** The voice cache is automatically cleared when a voice is updated or deleted, ensuring that the most up-to-date voice data is always used.


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

1.  Use the Docker image you built (`your-docker-username/chatterbox-tts:latest`) or the pre-built image from Docker Hub (`akashdeep000/chatterbox-tts:latest`).
2.  Configure a persistent volume and map it to `/app/voices` in the container.
3.  Expose the container's port `8000`.

This setup allows you to manage a scalable, high-performance TTS service with ease.