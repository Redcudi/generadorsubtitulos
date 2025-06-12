from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uuid
import os
import logging

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Logging
logging.basicConfig(level=logging.INFO)

# Whisper model (tiny for fast deployment)
model = WhisperModel("tiny", compute_type="int8", cpu_threads=4)

@app.get("/")
def root():
    return {"message": "API activa para generar subtítulos"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporal
        temp_filename = f"{uuid.uuid4()}.mp4"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())

        # Transcripción
        segments, _ = model.transcribe(temp_filename, beam_size=5)

        # Generar TXT y SRT
        base_name = temp_filename.replace(".mp4", "")
        txt_path = f"{base_name}.txt"
        srt_path = f"{base_name}.srt"

        with open(txt_path, "w") as txt_file:
            for segment in segments:
                txt_file.write(segment.text.strip() + " ")

        with open(srt_path, "w") as srt_file:
            for i, segment in enumerate(segments, start=1):
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                srt_file.write(f"{i}\n{start} --> {end}\n{segment.text.strip()}\n\n")

        # Leer contenido para respuesta
        with open(txt_path, "r") as txt_file:
            txt_content = txt_file.read()

        with open(srt_path, "r") as srt_file:
            srt_content = srt_file.read()

        # Eliminar archivos temporales
        os.remove(temp_filename)
        os.remove(txt_path)
        os.remove(srt_path)

        return {
            "transcription": txt_content.strip(),
            "srt": srt_content.strip()
        }

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return {"error": "No se pudo procesar el video", "detail": str(e)}

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"