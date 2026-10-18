import streamlit as st
import torch
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
import tempfile, os
import numpy as np

st.set_page_config(page_title="V100 BEST META", layout="centered")
st.title("V100 BEST - Demucs META")
st.caption("htdemucs - tira voz da base 100%")

file = st.file_uploader("Carrega a musica", type=["mp3","wav","flac","m4a","mp4"])
if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    st.audio(file)
    
    if st.button("SEPARAR BEST - VOZ LIMPA", type="primary"):
        with st.spinner("Rodando Demucs BEST... primeira vez demora 1-2min"):
            # carrega com soundfile (sem bug do torchcodec)
            data, sr = sf.read(tmp_path, always_2d=True)
            wav = torch.from_numpy(data.T).float()  # [channels, samples]
            if wav.shape[0] == 1:
                wav = wav.repeat(2,1)
            if wav.shape[0] > 2:
                wav = wav[:2]
            
            model = get_model("htdemucs")
            model.eval()
            wav = wav.unsqueeze(0)  # [1,2,samples]
            
            with torch.no_grad():
                sources = apply_model(model, wav, device="cpu")[0]  # [4,2,samples]
            
            names = ["bateria","baixo","outros","voz"]
            for i, name in enumerate(names):
                out = sources[i].cpu().numpy().T  # [samples,2]
                out_path = f"/tmp/{name}.wav"
                sf.write(out_path, out, sr)
                st.success(f"{name.upper()} PRONTO")
                st.audio(out_path)
                with open(out_path, "rb") as f:
                    st.download_button(f"⬇️ Baixar {name}.wav", f, file_name=f"{name}.wav", key=name)
        os.unlink(tmp_path)
