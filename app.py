import os
import socketio
from fastapi import FastAPI, UploadFile, File
import shutil, uuid, os
from ml import evaluate

sio = socketio.AsyncServer(cors_allowed_origins="*")
app = FastAPI()
sio_app = socketio.ASGIApp(sio, app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@sio.event
async def join(sid, room):
    await sio.enter_room(sid, room)
    await sio.emit("user_joined", sid, room=room)

@sio.event
async def disconnect(sid):
    pass

@app.post("/evaluate")
async def evaluate_audio(audio: UploadFile = File(...)):
    fname = f"{UPLOAD_DIR}/{uuid.uuid4()}.wav"
    with open(fname, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    result = evaluate(fname)
    os.remove(fname)
    return result
app = sio_app
