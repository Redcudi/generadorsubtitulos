from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uuid
import os

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
    return {"message": "API activa para video a texto"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        filename = f"{uuid.uuid4()}_{file.filename}"
        with open(filename, "wb") as f:
            f.write(await file.read())

        segments, _ = model.transcribe(filename, beam_size=5)
        transcription = "\n".join([segment.text for segment in segments])

        os.remove(filename)
        return {"text": transcription.strip()}
    
    except Exception as e:
        return {"error": str(e)}
