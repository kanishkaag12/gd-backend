import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from livekit import AccessToken, VideoGrant

from ml import evaluate

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/token")
def get_token(room: str):
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(str(uuid.uuid4()))
        .with_grants(VideoGrant(room_join=True, room=room))
    )
    return {"token": token.to_jwt()}

@app.post("/evaluate")
async def evaluate_audio(audio: UploadFile = File(...)):
    path = f"temp_{uuid.uuid4()}.wav"
    with open(path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    result = evaluate(path)
    os.remove(path)
    return result
