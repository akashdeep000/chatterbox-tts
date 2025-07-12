import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import logging
import time
from typing import Optional

from .config import settings
from .dependencies import get_tts_engine
from .tts import TextToSpeechEngine
from .voice_manager import VoiceManager
from fastapi import File, UploadFile

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(debug=settings.DEBUG)

# --- Static Files ---
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Middleware ---

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Handled request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.4f}s")
    return response


@app.get("/", response_class=FileResponse)
async def read_root():
    return "static/index.html"


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the application is running.
    """
    return {"status": "ok"}

# --- Security ---
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query_scheme = APIKeyQuery(name="api_key", auto_error=False)

async def get_api_key(
    api_key_header: Optional[str] = Depends(api_key_header_scheme),
    api_key_query: Optional[str] = Depends(api_key_query_scheme),
):
    """
    Get API key from header or query string.
    """
    api_key = api_key_header or api_key_query

    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None

@app.api_route("/tts/generate", methods=["GET", "POST"], dependencies=[Depends(get_api_key)])
async def tts_generate(request: Request, tts_engine: TextToSpeechEngine = Depends(get_tts_engine)):
    """
    Generates and streams audio for the given text.
    Accepts both GET and POST requests.
    """
    if request.method == "POST":
        try:
            body = await request.json()
            text = body.get("text")
            voice_id = body.get("voice_id")
            chunk_size = body.get("chunk_size")
        except Exception:
            return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)
    else:  # GET request
        text = request.query_params.get("text")
        voice_id = request.query_params.get("voice_id")

    if not text:
        return JSONResponse(content={"error": "Text is required"}, status_code=400)

    try:
        logger.info(f"Received TTS request for voice_id: {voice_id}")
        audio_stream = tts_engine.stream(text=text, voice_id=voice_id)
        return StreamingResponse(audio_stream, media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


@app.post("/voices", dependencies=[Depends(get_api_key)])
async def upload_voice(
    file: UploadFile = File(...),
    voice_manager: VoiceManager = Depends(),
    tts_engine: TextToSpeechEngine = Depends(get_tts_engine)
):
    """
    Upload a new voice file.
    """
    try:
        contents = await file.read()
        voice_manager.save_voice(file.filename, contents)
        tts_engine.clear_voice_cache(file.filename)  # Invalidate cache
        return JSONResponse(content={"voice_id": file.filename, "message": "Voice uploaded successfully."}, status_code=201)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Voice upload failed: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

@app.get("/voices", dependencies=[Depends(get_api_key)])
async def list_voices(voice_manager: VoiceManager = Depends()):
    """
    Get a list of all available voices.
    """
    return voice_manager.list_voices()

@app.delete("/voices/{voice_id}", dependencies=[Depends(get_api_key)])
async def delete_voice(
    voice_id: str,
    voice_manager: VoiceManager = Depends(),
    tts_engine: TextToSpeechEngine = Depends(get_tts_engine)
):
    """
    Delete a specific voice by its ID.
    """
    try:
        voice_manager.delete_voice(voice_id)
        tts_engine.clear_voice_cache(voice_id)  # Invalidate cache
        return JSONResponse(content={"message": f"Voice '{voice_id}' deleted successfully."})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found.")
    except Exception as e:
        logger.error(f"Voice deletion failed: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)
