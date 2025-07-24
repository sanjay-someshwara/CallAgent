from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uvicorn
import torch
import numpy as np
import librosa
from ultravox.inference.ultravox_infer import UltravoxInference
from ultravox.data.data_sample import VoiceSample
from typing import Dict, Any
from uuid import uuid4

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# Session storage for conversation history
session_histories: Dict[str, Dict[str, Any]] = {}

# --- Ultravox Model Setup ---
ULTRAVOX_MODEL_PATH = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
ULTRAVOX_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ULTRAVOX_DATA_TYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

from enum import Enum
class WhisperModel(str, Enum):
    TINY = "openai/whisper-tiny"
    BASE = "openai/whisper-base"
    SMALL = "openai/whisper-small"
    MEDIUM = "openai/whisper-medium"
    LARGE = "openai/whisper-large"
    LARGE_V2 = "openai/whisper-large-v2"
    LARGE_V3 = "openai/whisper-large-v3"
    LARGE_V3_TURBO = "openai/whisper-large-v3-turbo"

ultravox_pipeline = UltravoxInference(
    model_path=ULTRAVOX_MODEL_PATH,
    device=ULTRAVOX_DEVICE,
    data_type=ULTRAVOX_DATA_TYPE,
    conversation_mode=True,
    audio_processor_id=WhisperModel.LARGE_V3_TURBO.value
)

@app.get("/models")
async def get_models():
    return {
        "models": [model.value for model in WhisperModel]
    }

from pydantic import BaseModel

class ModelRequest(BaseModel):
    model: WhisperModel

@app.post("/setmodel")
async def set_model(request: ModelRequest):
    global ultravox_pipeline, session_histories
    # Clear all conversation histories when model changes
    session_histories.clear()
    
    ultravox_pipeline = UltravoxInference(
        model_path=ULTRAVOX_MODEL_PATH,
        device=ULTRAVOX_DEVICE,
        data_type=ULTRAVOX_DATA_TYPE,
        conversation_mode=True,
        audio_processor_id=request.model.value
    )
    return {
        "message": f"Model set to {request.model.value}. Conversation history cleared."
    }

@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    prompt: str = "<|audio|>",
    session_id: str = Query(None, description="Session ID for conversation history")
):
    # Generate a session_id if not provided
    if not session_id:
        session_id = str(uuid4())

    # Retrieve session history
    history = session_histories.get(session_id, {})
    past_messages = history.get("past_messages", [])
    past_key_values = history.get("past_key_values", None)

    # Save uploaded audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()

        # Load audio for Ultravox
        audio, sr = librosa.load(tmp.name, sr=16000)
        audio = audio.astype(np.float32)

    # Create VoiceSample for Ultravox
    messages = [{"role": "user", "content": prompt}]
    sample = VoiceSample(messages=messages, audio=audio, sample_rate=sr)

    # Set conversation history in the pipeline
    ultravox_pipeline.update_conversation(past_messages, past_key_values)

    # Run Ultravox inference
    output = ultravox_pipeline.infer(sample)

    # Update session history
    session_histories[session_id] = {
        "past_messages": ultravox_pipeline.past_messages,
        "past_key_values": ultravox_pipeline.past_key_values,
    }

    return {
        "session_id": session_id,
        "ultravox_response": output.text
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)