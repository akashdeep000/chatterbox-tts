import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
from typing import Optional

# Placeholder for TTS engine and voice management
# These will be implemented in other files.
from tts import TextToSpeechEngine

app = FastAPI()

# Initialize TTS engine and voice manager
# In a real app, these would be initialized with proper configuration
tts_engine = TextToSpeechEngine()

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None

@app.post("/tts/generate")
async def tts_generate(request: TTSRequest):
    """
    Generates a complete audio file for the given text.
    """
    try:
        # Generate audio
        audio_data = tts_engine.generate(text=request.text, voice_id=request.voice_id)

        # Return the audio as a streaming response
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}, 500

@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket):
    """
    Streams generated audio in chunks over a WebSocket connection.
    """
    await websocket.accept()
    try:
        while True:
            # Receive a JSON request from the client
            data = await websocket.receive_json()
            text = data.get("text")
            voice_id = data.get("voice_id")

            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue

            # Stream audio chunks
            for chunk in tts_engine.stream(text=text, voice_id=voice_id):
                await websocket.send_bytes(chunk)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)