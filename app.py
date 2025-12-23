import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from livekit.api import AccessToken, RoomGrant

from ml import evaluate

# -----------------------------
# Environment variables
# -----------------------------
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    raise RuntimeError("LIVEKIT_API_KEY or LIVEKIT_API_SECRET not set")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="GD Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LiveKit token endpoint
# -----------------------------
@app.get("/token")
def get_token(room: str):
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(str(uuid.uuid4()))
        .with_grants(
            RoomGrant(
                room_join=True,
                room=room
            )
        )
    )
    return {"token": token.to_jwt()}

# -----------------------------
# ML Evaluation endpoint
# -----------------------------
@app.post("/evaluate")
async def evaluate_audio(audio: UploadFile = File(...)):
    path = f"/tmp/{uuid.uuid4()}.wav"

    with open(path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    result = evaluate(path)

    os.remove(path)
    return result
