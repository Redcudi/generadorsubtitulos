from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os
import logging

app = FastAPI()

# CORS configw
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Logger
logging.basicConfig(level=logging.INFO)

# Modelo
model = WhisperModel("tiny", compute_type="int8", cpu_threads=4)

@app.get("/")
def root():
    return {"message": "API activa para generación de subtítulos (.txt y .srt)"}

@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    try:
        logging.info(f"📥 Recibido: {file.filename}")

        # Guardar temporalmente el archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name

        logging.info(f"🧠 Procesando con Whisper: {temp_video_path}")

        segments, _ = model.transcribe(temp_video_path, beam_size=5)
        transcription_txt = []
        transcription_srt = []
        count = 1

        for segment in segments:
            # TXT
            transcription_txt.append(segment.text)

            # SRT
            start = segment.start
            end = segment.end

            def format_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                ms = int((seconds - int(seconds)) * 1000)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"

            transcription_srt.append(
                f"{count}\n{format_time(start)} --> {format_time(end)}\n{segment.text}\n"
            )
            count += 1

        return {
            "transcription": " ".join(transcription_txt).strip(),
            "srt": "\n".join(transcription_srt).strip()
        }

    except Exception as e:
        logging.error(f"❌ Error durante la transcripción: {e}")
        return {"error": "No se pudo transcribir el video", "detail": str(e)}
    finally:
        # Eliminar el video aunque haya error
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)