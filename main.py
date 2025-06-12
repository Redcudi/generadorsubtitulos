from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uuid
import os
import logging

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)

# Cargar modelo
model = WhisperModel("tiny", compute_type="int8", cpu_threads=4)

def generate_srt(segments):
    srt_output = []
    for i, segment in enumerate(segments, start=1):
        start = segment.start
        end = segment.end

        def format_time(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        srt_output.append(f"{i}")
        srt_output.append(f"{format_time(start)} --> {format_time(end)}")
        srt_output.append(segment.text.strip())
        srt_output.append("")  # Línea en blanco

    return "\n".join(srt_output)

@app.get("/")
def root():
    return {"message": "API activa para generar subtítulos"}

@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporal
        file_id = str(uuid.uuid4())
        temp_path = f"{file_id}.mp4"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        logging.info(f"Procesando archivo: {temp_path}")

        # Transcribir
        segments, _ = model.transcribe(temp_path, beam_size=5)
        transcription = " ".join([seg.text for seg in segments])
        srt_text = generate_srt(segments)

        # Eliminar archivo temporal
        os.remove(temp_path)

        return {
            "transcription": transcription.strip(),
            "srt": srt_text.strip()
        }

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return {"error": "No se pudo transcribir", "detail": str(e)}