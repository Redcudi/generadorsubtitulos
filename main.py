from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = WhisperModel("tiny", compute_type="int8", cpu_threads=4)

@app.get("/")
def root():
    return {"message": "API activa"}

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        ext = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"

        with open(filename, "wb") as buffer:
            buffer.write(await file.read())

        segments, _ = model.transcribe(filename)
        transcription = " ".join([segment.text for segment in segments])

        os.remove(filename)
        return {"transcription": transcription.strip()}

    except Exception as e:
        return {"error": "No se pudo transcribir", "detail": str(e)}