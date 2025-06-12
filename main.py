from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import os
import uuid
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

logging.basicConfig(level=logging.INFO)
model = WhisperModel("tiny", compute_type="int8", cpu_threads=4)

@app.get("/")
def root():
    return {"message": "API activa para generar subtítulos"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        temp_path = f"{file_id}.mp4"

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        segments_gen, _ = model.transcribe(temp_path, beam_size=5)
        segments = list(segments_gen)

        transcription_txt = " ".join([seg.text for seg in segments])

        # SRT generation
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            srt_lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")

        transcription_srt = "\n".join(srt_lines)

        os.remove(temp_path)

        return {
            "transcription": transcription_txt.strip(),
            "srt": transcription_srt.strip()
        }

    except Exception as e:
        logging.error(f"❌ Error en transcripción: {e}")
        return {"error": "No se pudo transcribir", "detail": str(e)}

def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"