import uuid
import shutil
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ml import evaluate  # your existing ML logic

app = FastAPI(title="GD Evaluation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluate")
async def evaluate_audio(audio: UploadFile = File(...)):
    path = f"/tmp/{uuid.uuid4()}.wav"

    with open(path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    result = evaluate(path)
    os.remove(path)
    return result
