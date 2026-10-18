import streamlit as st
import torch
import torchaudio
import tempfile
import os
import zipfile
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model

st.set_page_config(page_title="9flu LITE - Demucs", layout="centered")
st.title("9flu - Separador LITE")
st.caption("Modo leve para não tomar limite de CPU do Streamlit")

@st.cache_resource(show_spinner="Carregando modelo (só na primeira vez)...")
def load_model():
    model = get_model("htdemucs")
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar modelo: {e}")
    st.stop()

uploaded = st.file_uploader("Suba MP3 ou WAV (até 3 min recomendado no free)", type=["mp3","wav","flac"])

if uploaded:
    # salva temporário
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded.name)
        with open(input_path, "wb") as f:
            f.write(uploaded.read())
        
        st.audio(input_path)
        
        if st.button("🔥 Separar Voz e Instrumental", type="primary"):
            try:
                # Carrega com torchaudio (sem torchcodec)
                wav, sr = torchaudio.load(input_path)
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                
                # Resample se precisar
                if sr != model.samplerate:
                    resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                    wav = resampler(wav)
                
                st.info("Separando... isso leva ~1 min, não fecha a aba.")
                progress = st.progress(0)
                
                # MODO LITE: split=True evita estourar CPU/RAM
                with torch.no_grad():
                    # overlap menor = mais leve
                    sources = apply_model(
                        model, 
                        wav[None], 
                        device="cpu", 
                        split=True, 
                        overlap=0.25,
                        progress=True
                    )[0]
                
                progress.progress(100)
                
                # sources: drums, bass, other, vocals
                # Vamos juntar tudo que não é vocal como "instrumental"
                vocals = sources[3]
                instrumental = sources[0] + sources[1] + sources[2]
                
                # Salva resultados
                out_dir = Path(tmpdir) / "saida"
                out_dir.mkdir(exist_ok=True)
                
                vocal_path = out_dir / "vocal.wav"
                inst_path = out_dir / "instrumental.wav"
                
                torchaudio.save(str(vocal_path), vocals.cpu(), model.samplerate)
                torchaudio.save(str(inst_path), instrumental.cpu(), model.samplerate)
                
                st.success("Pronto!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Vocal")
                    st.audio(str(vocal_path))
                    with open(vocal_path, "rb") as f:
                        st.download_button("Baixar Vocal", f, file_name="vocal.wav")
                with col2:
                    st.subheader("Instrumental")
                    st.audio(str(inst_path))
                    with open(inst_path, "rb") as f:
                        st.download_button("Baixar Instrumental", f, file_name="instrumental.wav")
                
            except Exception as e:
                st.error(f"Deu erro: {e}")
                st.write("Tenta um áudio menor (corta para 2 min)")

st.divider()
st.write("Dica: Se der amarelo de novo, diminua o áudio para menos de 2 minutos.")
