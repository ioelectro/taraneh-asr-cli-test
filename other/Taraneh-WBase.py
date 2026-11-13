
import soundfile as sf
import numpy as np
import torch

from scipy.signal import resample_poly
from transformers import pipeline

MODEL_ID = "areffarhadi/Taraneh-WBase"

print("Loading Taraneh-WBase on CPU...")

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device="cpu",
    torch_dtype=torch.float32,
)

print("Model loaded.")

# Test with a WAV file
audio, sr = sf.read("audio/test.wav")

if audio.ndim == 2:
    audio = audio.mean(axis=1)

if sr != 16000:
    audio = resample_poly(audio, 16000, sr)
    sr = 16000

result = asr(
    {"raw": audio.astype(np.float32), "sampling_rate": sr},
    generate_kwargs={
        "language": "persian",
        "task": "transcribe",
    },
)

print("Recognized text:", result["text"])

