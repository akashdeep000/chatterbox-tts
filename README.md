# Chatterbox TTS

Chatterbox TTS is a high-performance, containerized Text-to-Speech (TTS) service designed for real-time audio generation and streaming. It supports custom voice cloning and is optimized for deployment on GPU-accelerated platforms like RunPod.

## Project Overview

This project provides a flexible and scalable solution for generating speech from text. It exposes a RESTful API for generating complete audio files and is built to be easily deployable and managed as a serverless endpoint.

### High-Level Architecture

The following diagram illustrates the high-level architecture of the Chatterbox TTS service:

```mermaid
graph TD
    subgraph "Client"
        A[User Application]
    end

    subgraph "Chatterbox TTS Service"
        B[FastAPI Web Server]
        C[TTS Engine]
        D[Voice Manager]
        E[Voice Cache]
    end

    subgraph "Infrastructure"
        F[Docker Container]
        G[GPU]
        H[Persistent Volume]
    end

    A -- "HTTP Request /tts/generate" --> B
    B -- "Calls" --> C
    C -- "Uses" --> G
    C -- "Requests Voice Conditionals" --> D
    D -- "Loads/Saves Voices" --> H
    D -- "Caches Conditionals" --> E
    C -- "Uses Cached Conditionals" --> E
    B -- "Streams Audio Response" --> A

    style F fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    style H fill:#cfc,stroke:#333,stroke-width:2px
```

## Prerequisites

To run this project, you will need:

*   **Docker**: The application is containerized and requires Docker to be installed.
*   **GPU**: A CUDA-enabled GPU is necessary for the TTS model to perform efficiently.

## Configuration

Before running the application, you must configure the following environment variables. You can set them directly in your shell or create a `.env` file in the project root.

### Application Configuration Parameters

These parameters are for the overall application settings.

*   `HOST`: The host address for the application server. Defaults to `0.0.0.0`.
*   `PORT`: The port for the application server. Defaults to `8000`.
*   `DEBUG`: Enable or disable debug mode for the application. Defaults to `False`.
*   `LOG_LEVEL`: The logging level (e.g., `INFO`, `DEBUG`, `WARNING`, `ERROR`). Defaults to `INFO`.
*   `VOICES_DIR`: Directory where custom voices are stored. Defaults to `voices/`.
*   `PRELOADED_VOICES_DIR`: Directory for preloaded voices. Defaults to `preloaded-voices/`.
*   `MODEL_PATH`: Path to the directory containing TTS models. Defaults to `models`.
*   `API_KEY`: **(Required)** Your secret API key for securing the service.
*   `CORS_ORIGINS`: A comma-separated list of allowed origins (e.g., `"http://localhost:3000,https://your-frontend.com"`). Defaults to `*` (all origins).

### TTS Configuration Parameters

These parameters control the Text-to-Speech engine's behavior and can be set via environment variables prefixed with `TTS_`.

*   `TTS_VOICE_EXAGGERATION_FACTOR`: Controls the voice_exaggeration_factor of the voice. Higher values increase expressiveness. Defaults to `0.5`.
*   `TTS_CFG_GUIDANCE_WEIGHT`: Classifier-free guidance weight. Influences how strongly the model adheres to the text prompt. Defaults to `0.5`.
*   `TTS_SYNTHESIS_TEMPERATURE`: synthesis_temperature for text-to-speech synthesis. Higher values lead to more varied but potentially less coherent output. Defaults to `0.8`.
*   `TTS_TEXT_PROCESSING_CHUNK_SIZE`: Maximum number of characters per text chunk for processing. Defaults to `100`.
*   `TTS_AUDIO_TOKENS_PER_SLICE`: Number of audio tokens per slice during streaming synthesis. Defaults to `35`.
*   `TTS_REMOVE_LEADING_MILLISECONDS`: Duration in milliseconds to remove from the start of generated audio. Defaults to `0`.
*   `TTS_REMOVE_TRAILING_MILLISECONDS`: Duration in milliseconds to remove from the end of generated audio. Defaults to `0`.
*   `TTS_CHUNK_OVERLAP_STRATEGY`: Strategy for overlapping audio chunks: `"full"` (overlap and crossfade) or `"zero"` (no overlap). Defaults to `"full"`.
*   `TTS_CROSSFADE_DURATION_MILLISECONDS`: Duration in milliseconds for crossfading between audio chunks. Defaults to `8`.

### Example `.env` file:

```
API_KEY="your-super-secret-api-key"
CORS_ORIGINS="http://localhost:3000,https://your-app.com"
LOG_LEVEL="DEBUG"

# TTS Configuration
TTS_VOICE_EXAGGERATION_FACTOR=0.7
TTS_SYNTHESIS_TEMPERATURE=0.9
TTS_CHUNK_OVERLAP_STRATEGY="zero"
TTS_CROSSFADE_DURATION_MILLISECONDS=10
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

#### TTS Parameters

The `/tts/generate` endpoint accepts several parameters to customize the audio generation. These can be provided in the query string for `GET` requests or in the JSON body for `POST` requests.

| Parameter                   | Type    | Default Value      | Description                                                                                                                            |
| --------------------------- | ------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `text`                      | string  | **(Required)**     | The text to be converted to speech.                                                                                                    |
| `voice_id`                  | string  | `None`             | The ID of the custom voice to use (e.g., `your_voice.wav`). If not provided, a default voice is used.                                    |
| `~~voice_exaggeration_factor~~`          | float   | `TTS_VOICE_EXAGGERATION_FACTOR` | Controls the emotional expressiveness of the speech. Higher values produce more exaggerated speech.                                    |
| `cfg_guidance_weight`                | float   | `TTS_CFG_GUIDANCE_WEIGHT`       | Classifier-Free Guidance weight. Higher values make the speech more closely follow the text, but can reduce naturalness.               |
| `synthesis_temperature`               | float   | `TTS_SYNTHESIS_TEMPERATURE`     | Controls the randomness of the output. Higher values produce more varied and creative speech, while lower values are more deterministic. |
| `text_processing_chunk_size`           | integer | `TTS_TEXT_PROCESSING_CHUNK_SIZE`| The number of characters to process in each text chunk. Smaller values can reduce latency but may affect prosody.                      |
| `audio_tokens_per_slice`          | integer | `TTS_AUDIO_TOKENS_PER_SLICE`    | The number of audio tokens to generate in each slice. This affects the granularity of the streaming output.                            |
| `remove_trailing_milliseconds`       | integer | `TTS_REMOVE_TRAILING_MILLISECONDS`| The number of milliseconds to trim from the end of the final audio chunk.                        |
| `remove_leading_milliseconds` | integer | `TTS_REMOVE_LEADING_MILLISECONDS`| The number of milliseconds to trim from the beginning of the first audio chunk.                                                        |
| `chunk_overlap_method`      | string  | `TTS_CHUNK_OVERLAP_STRATEGY`    | The method for handling overlapping audio chunks. Can be `"full"` or `"zero"`.                                                         |
| `crossfade_duration_milliseconds`        | integer   | `TTS_CROSSFADE_DURATION_MILLISECONDS`| The duration of the crossfade between audio chunks in milliseconds.

### Performance Optimizations

The TTS engine is optimized for real-time performance through several mechanisms:

#### Parameter Tuning for Quality and Speed

The `text_processing_chunk_size`, `audio_tokens_per_slice`, and `chunk_overlap_method` parameters are crucial for balancing audio quality and streaming latency. Understanding how they work together allows you to fine-tune the TTS engine for your specific needs.

*   **`text_processing_chunk_size`**: This parameter determines how the input text is split into smaller pieces. The T3 model processes one chunk at a time.
    *   **Smaller values** (e.g., 50) lead to lower "time to first audio" because the first chunk is processed faster. However, this can sometimes result in less natural prosody, as the model has less context.
    *   **Larger values** (e.g., 200) provide more context to the model, which can improve the naturalness of the speech, but it will take longer to receive the first audio chunk.

*   **`audio_tokens_per_slice`**: After the T3 model converts a text chunk into a sequence of speech tokens, this parameter determines how many of those tokens are sent to the S3Gen model at a time to be converted into audio.
    *   **Smaller values** (e.g., 20) result in smaller, more frequent audio chunks being streamed to the client, which can create a smoother streaming experience.
    *   **Larger values** (e.g., 50) will result in fewer, larger audio chunks, which can be more efficient but may feel less "real-time."

*   **`chunk_overlap_method`**: This parameter defines how the audio from different text chunks is stitched together.
    *   **`"full"`**: This method creates a seamless overlap between audio chunks, which generally produces the highest quality audio by avoiding clicks or pauses. It is slightly more computationally intensive.
    *   **`"zero"`**: This method simply concatenates the audio chunks. It is faster but may occasionally produce audible artifacts at the seams between chunks.

*   **`crossfade_duration_milliseconds`**: This parameter determines the duration of the crossfade between audio chunks.

The following diagram illustrates how these parameters relate to each other in the TTS process:

```mermaid
graph TD
    subgraph "Input Text"
        A["The quick brown fox jumps over the lazy dog."]
    end

    subgraph "Text Chunking (text_processing_chunk_size)"
        B["The quick brown fox..."]
        C["...jumps over the lazy dog."]
    end

    subgraph "T3 Model (Text-to-Tokens)"
        D["[Speech Tokens for Chunk 1]"]
        E["[Speech Tokens for Chunk 2]"]
    end

    subgraph "Token Slicing (audio_tokens_per_slice)"
        F["Slice 1.1"]
        G["Slice 1.2"]
        H["Slice 2.1"]
        I["Slice 2.2"]
    end

    subgraph "S3Gen Model (Tokens-to-Audio)"
        J["Audio for Slice 1.1"]
        K["Audio for Slice 1.2"]
        L["Audio for Slice 2.1"]
        M["Audio for Slice 2.2"]
    end

    subgraph "Output Stream (chunk_overlap_method)"
        N["Final Audio Stream"]
    end

    A -- "split by text_processing_chunk_size" --> B & C
    B --> D
    C --> E
    D -- "split by audio_tokens_per_slice" --> F & G
    E -- "split by audio_tokens_per_slice" --> H & I
    F --> J
    G --> K
    H --> L
    I --> M
    J & K & L & M -- "stitched by chunk_overlap_method" --> N

```

*   **Aggressive Pre-caching**: To eliminate warm-up latency, the service pre-loads all TTS models and pre-caches the conditioning data for *all available voices* into GPU memory at startup. This ensures that every voice is ready for immediate, high-performance inference from the very first request.
*   **Intelligent Cache Invalidation**: The voice cache is automatically and precisely invalidated when a voice is updated or deleted, guaranteeing that the system always uses the most recent voice data without requiring a manual restart.
*   **Asynchronous Streaming Pipeline**: The core of the real-time streaming is a highly efficient producer-consumer pattern. The T3 model (producer) generates speech tokens concurrently while the S3Gen model (consumer) converts them into audio. This decoupling prevents stalls and ensures a smooth, continuous flow of audio data.
*   **Proactive Inference**: The pipeline uses a signaling mechanism that allows the T3 model to proactively start processing the *next* chunk of text while the S3Gen model is still working on the current one. This advanced optimization minimizes gaps in audio generation, leading to a significant reduction in perceived latency for longer texts.

#### Real-Time TTS Generation Sequence

The real-time audio streaming is achieved through a producer-consumer pattern, where the T3 model produces speech tokens and the S3Gen model consumes them to generate audio. This allows for a continuous stream of audio with low latency.

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant T3_Producer as T3 Producer
    participant S3Gen_Consumer as S3Gen Consumer
    participant SpeechTokenQueue as Speech Token Queue
    participant AudioChunkQueue as Audio Chunk Queue

    Client->>FastAPI: POST /tts/generate (text)
    FastAPI->>T3_Producer: Start Task (text_chunks)
    loop For each text chunk
        T3_Producer->>T3_Producer: Generate Speech Tokens
        T3_Producer->>SpeechTokenQueue: Enqueue Speech Tokens
    end
    FastAPI->>S3Gen_Consumer: Start Task
    loop Until stream ends
        S3Gen_Consumer->>SpeechTokenQueue: Dequeue Speech Tokens
        S3Gen_Consumer->>S3Gen_Consumer: Generate Audio Chunk
        S3Gen_Consumer->>AudioChunkQueue: Enqueue Audio Chunk
        AudioChunkQueue-->>FastAPI: Dequeue Audio Chunk
        FastAPI-->>Client: Stream Audio Chunk
    end
```

### Voice Management API

The service provides a set of RESTful endpoints to manage custom voices.

#### Voice Caching and Management

The voice management system is designed for efficiency and scalability. When a voice is uploaded, it is stored persistently. To ensure the lowest possible latency, the application automatically pre-caches all available voices into memory on startup. This means that all voices are ready for immediate use without any warm-up delay on the first request. The cache is also intelligently invalidated and updated whenever a voice is uploaded or deleted.

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