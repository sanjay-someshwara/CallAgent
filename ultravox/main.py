import sys
import os
import io
import asyncio
import numpy as np
import soundfile as sf
from typing import Optional
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from optimum.bettertransformer import BetterTransformer

from ultravox.inference.ultravox_infer import UltravoxInference
from ultravox import data as datasets
from ultravox.inference import base

# --- Configuration ---
os.environ.setdefault("HF_TOKEN", "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")
MODEL_PATH = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_TYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

WHISPER_MODEL_ID = "openai/whisper-tiny"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_DTYPE = torch.float16 if WHISPER_DEVICE == "cuda" else torch.float32

SAMPLE_RATE = 16000

# --- Global state ---
uv_inference_pipeline: Optional[UltravoxInference] = None
whisper_processor = None
whisper_model = None

# Per-user session store
user_sessions = defaultdict(lambda: {
    "chat_history": [],
    "last_audio": np.array([]),
    "current_task": None
})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global uv_inference_pipeline, whisper_processor, whisper_model
    print("Starting server...")

    try:
        whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_ID,
            torch_dtype=WHISPER_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(WHISPER_DEVICE)

    except Exception as e:
        print(f"Failed to load Whisper: {e}")
        sys.exit(1)

    try:
        uv_inference_pipeline = UltravoxInference(
            model_path=MODEL_PATH,
            device=DEVICE,
            data_type=DATA_TYPE,
            conversation_mode=True
        )
    except Exception as e:
        print(f"Failed to load Ultravox: {e}")
        sys.exit(1)

    yield

    whisper_processor = whisper_model = uv_inference_pipeline = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="Interruptable Ultravox Voice Bot", lifespan=lifespan)


def transcribe_audio(audio_data: np.ndarray) -> str:
    if whisper_model is None or whisper_processor is None:
        raise RuntimeError("Whisper model not available")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    input_features = whisper_processor(
        audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(WHISPER_DTYPE).to(WHISPER_DEVICE)

    predicted_ids = whisper_model.generate(
        input_features,
        forced_decoder_ids=whisper_processor.get_decoder_prompt_ids(language="english", task="transcribe")
    )

    return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def get_ultravox_response(transcribed_text: str, audio_data: np.ndarray, messages: list) -> str:
    sample = datasets.VoiceSample(
        messages=messages + [{"role": "user", "content": f"{transcribed_text} <|audio|>"}],
        audio=audio_data,
        sample_rate=SAMPLE_RATE
    )

    response_text = ""
    for chunk in uv_inference_pipeline.infer_stream(sample):
        if isinstance(chunk, base.InferenceChunk) and chunk.text:
            response_text += chunk.text

    return response_text.strip()


@app.post("/process-audio/")
async def process_audio(request: Request, audio_file: UploadFile = File(...)):
    user_id = request.client.host  # Replace with token/session ID for production
    session = user_sessions[user_id]

    # Cancel ongoing task
    if session["current_task"] and not session["current_task"].done():
        session["current_task"].cancel()
        print(f"Interrupted previous task for {user_id}")

    # Read and parse audio
    try:
        audio_bytes = await audio_file.read()
        audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        if samplerate != SAMPLE_RATE:
            import resampy
            audio_data = resampy.resample(audio_data, samplerate, SAMPLE_RATE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e}")

    # Combine with previous audio
    if session["last_audio"].size > 0:
        audio_data = np.concatenate((session["last_audio"], audio_data))
    session["last_audio"] = audio_data

    # Define async task
    async def run_inference():
        transcribed = transcribe_audio(audio_data)
        session["chat_history"].append({"role": "user", "content": transcribed})

        response = get_ultravox_response(transcribed, audio_data, session["chat_history"])
        session["chat_history"].append({"role": "assistant", "content": response})

        return {
            "transcription": transcribed,
            "response": response,
            "chat_history": session["chat_history"]
        }

    # Run task
    session["current_task"] = asyncio.create_task(run_inference())
    try:
        result = await session["current_task"]
        return JSONResponse(content=result)
    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Request interrupted")

