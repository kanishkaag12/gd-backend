import os
import uuid
import shutil
import socketio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ml import evaluate

# -----------------------------
# Socket.IO setup
# -----------------------------
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    async_mode="asgi"
)

fastapi_app = FastAPI(title="GD Backend")

# -----------------------------
# CORS (MANDATORY)
# -----------------------------
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Attach Socket.IO to FastAPI
# -----------------------------
app = socketio.ASGIApp(sio, fastapi_app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Socket.IO events
# -----------------------------
@sio.event
async def connect(sid, environ):
    print("User connected:", sid)

@sio.event
async def join(sid, room):
    await sio.enter_room(sid, room)
    await sio.emit("user_joined", sid, room=room)

@sio.event
async def disconnect(sid):
    print("User disconnected:", sid)

# -----------------------------
# REST API for ML evaluation
# -----------------------------
@fastapi_app.post("/evaluate")
async def evaluate_audio(audio: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{uuid.uuid4()}.wav"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    result = evaluate(file_path)

    os.remove(file_path)
    return result
