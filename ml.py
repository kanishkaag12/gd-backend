import torchaudio
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# Load models once
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

TOPIC = "Impact of Artificial Intelligence on Education"
topic_emb = embedder.encode(TOPIC, convert_to_tensor=True)

def evaluate(audio_path):
    # Speech to text
    waveform, sr = torchaudio.load(audio_path)
    inputs = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt")
    ids = whisper.generate(inputs["input_features"])
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    # Content relevance
    emb = embedder.encode(text, convert_to_tensor=True)
    content = util.cos_sim(topic_emb, emb).item()

    # Fluency
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    fluency = min(tempo / 200, 1.0)

    score = round((0.6 * content + 0.4 * fluency) * 10, 2)

    return {
        "transcript": text,
        "content_score": round(content, 2),
        "fluency": round(fluency, 2),
        "final_score": score
    }
