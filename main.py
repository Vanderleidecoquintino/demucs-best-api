from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import subprocess, os, uuid

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/demucs")
async def demucs_separate(file: UploadFile):
    id = str(uuid.uuid4())[:8]
    input_path = f"/tmp/{id}_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # Roda Demucs htdemucs_ft - BEST qualidade
    subprocess.run(["demucs", "-n", "htdemucs_ft", "--mp3", "--mp3-bitrate", "320", "-o", f"/tmp/out_{id}", input_path])
    
    base = f"/tmp/out_{id}/htdemucs_ft/{id}_{file.filename.split('.')[0]}"
    return {
        "vocals": f"/download/{id}/vocals.mp3",
        "drums": f"/download/{id}/drums.mp3", 
        "bass": f"/download/{id}/bass.mp3",
        "other": f"/download/{id}/other.mp3"
    }

@app.get("/")
def ok(): return {"status":"Demucs BEST API ON - Motor V95 + Meta"}
