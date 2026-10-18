import streamlit as st
import torchaudio, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import tempfile

st.set_page_config(page_title="V100 BEST META", layout="centered")
st.title("V100 BEST - Demucs META")
st.caption("Motor htdemucs da Meta - voz 100% limpa")

file = st.file_uploader("Carrega a musica", type=["mp3","wav","flac","m4a"])
if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    st.audio(file)
    if st.button("SEPARAR BEST", type="primary"):
        with st.spinner("Rodando Demucs BEST..."):
            wav, sr = torchaudio.load(tmp_path)
            if wav.shape[0]==1: wav=wav.repeat(2,1)
            model = get_model("htdemucs")
            model.eval()
            wav = wav.unsqueeze(0)
            with torch.no_grad():
                sources = apply_model(model, wav, device="cpu")[0]
            names=["bateria","baixo","outros","voz"]
            for i,name in enumerate(names):
                out_path = f"/tmp/{name}.wav"
                torchaudio.save(out_path, sources[i], sr)
                st.success(f"{name.upper()} pronto")
                st.audio(out_path)
