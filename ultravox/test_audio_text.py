import os
import torch
import librosa
import numpy as np

# Import necessary classes from ultravox repo
from ultravox.inference.ultravox_infer import UltravoxInference
from ultravox import data as datasets # Need to import the datasets module for VoiceSample

os.environ["HF_TOKEN"] = "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" 

# --- Configuration ---
MODEL_PATH = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_TYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

print(f"Loading Ultravox model: {MODEL_PATH} on device: {DEVICE} with dtype: {DATA_TYPE}")

# Initialize the UltravoxInference class
inference_pipeline = UltravoxInference(
    model_path=MODEL_PATH,
    device=DEVICE,
    data_type=DATA_TYPE,
    conversation_mode=False # Set to True to enable conversation history
)

print("Model loaded successfully!")

# --- Inference Examples ---

# Example 1: Text-only input
print("\n--- Text-only example ---")
text_query_messages = [
    {"role": "user", "content": "What is the capital of France?"}
]
# Create a VoiceSample for text-only input
text_sample = datasets.VoiceSample(messages=text_query_messages)

# Use the infer method
text_output = inference_pipeline.infer(text_sample)
print(f"User: {text_query_messages[0]['content']}")
print(f"Assistant: {text_output.text}") # Access the generated text from VoiceOutput

# Example 2: Audio 
audio_file_path = "regional.wav"
try:
    # Load your audio file
    audio, sr = librosa.load(audio_file_path, sr=16000)
    audio = audio.astype(np.float32)

    # The <|audio|> token is crucial for multimodal input
    multimodal_messages = [
        {"role": "user", "content": f"<|audio|>"}
    ]

    # Create a VoiceSample for audio + text input
    multimodal_sample = datasets.VoiceSample(
        messages=multimodal_messages,
        audio=audio,
        sample_rate=sr
    )

    # Use the infer method
    multimodal_output = inference_pipeline.infer(multimodal_sample)
    print(f"User (with audio): {multimodal_messages[0]['content']}")
    print(f"Assistant: {multimodal_output.text}") # Access the generated text

except FileNotFoundError:
    print(f"\nError: Audio file not found at '{audio_file_path}'. Please provide a valid path or ensure the dummy file is created.")
except Exception as e:
    print(f"\nAn error occurred during audio processing: {e}")