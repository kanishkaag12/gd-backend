import torchaudio
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# -------- Load models once --------
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

TOPIC = "Impact of Artificial Intelligence on Education"
topic_embedding = embedder.encode(TOPIC, convert_to_tensor=True)

# -------- Main evaluation function --------
def evaluate(audio_path: str):
    # 1️⃣ Speech to text
    waveform, sr = torchaudio.load(audio_path)
    inputs = whisper_processor(
        waveform.squeeze(),
        sampling_rate=sr,
        return_tensors="pt"
    )
    ids = whisper_model.generate(inputs["input_features"])
    transcript = whisper_processor.batch_decode(
        ids, skip_special_tokens=True
    )[0]

    # 2️⃣ Topic relevance
    text_embedding = embedder.encode(transcript, convert_to_tensor=True)
    relevance = util.cos_sim(topic_embedding, text_embedding).item()

    # 3️⃣ Fluency & confidence (audio-based)
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    pauses = librosa.effects.split(y, top_db=20)
    spoken = sum(p[1] - p[0] for p in pauses)
    pause_ratio = 1 - (spoken / len(y))

    fluency = min(tempo / 200, 1.0)
    confidence = 1 - pause_ratio

    # 4️⃣ Final score
    final_score = round(
        (0.5 * relevance + 0.3 * fluency + 0.2 * confidence) * 10,
        2
    )

    return {
        "transcript": transcript,
        "scores": {
            "topic_relevance": round(relevance, 2),
            "fluency": round(fluency, 2),
            "confidence": round(confidence, 2),
        },
        "final_score": final_score,
        "on_topic": relevance >= 0.4
    }
