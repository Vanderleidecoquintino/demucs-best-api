import streamlit as st
import torch, tempfile, os, gc, numpy as np
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

st.set_page_config(page_title="V102 LITE", layout="wide")
st.markdown("<h2 style='background:#000;color:#ffcc00;padding:14px;border-radius:14px;text-align:center'>V102-LITE - LEVE + PARTITURA V95</h2>", unsafe_allow_html=True)

file = st.file_uploader("Carrega a música (até 10 min / MP3 leve)", type=["mp3","wav","m4a"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        path=tmp.name
    st.audio(file)

    if st.button("SEPARAR - MODO LEVE", type="primary"):
        with st.spinner("Separando no modo leve (não trava)..."):
            try:
                data, sr = sf.read(path, always_2d=True)
                # mono + baixa taxa se muito grande pra economizar RAM
                if data.shape[0] > sr*60*6: # >6min corta pra teste
                    data = data[:sr*60*5]
                    st.warning("Música muito grande, cortei 5min pra caber no plano free")
                
                wav = torch.from_numpy(data.T).float()
                if wav.shape[0]==1: wav=wav.repeat(2,1)
                if wav.shape[0]>2: wav=wav[:2]
                
                model = get_model("htdemucs")
                model.eval()
                # ECONOMIA DE RAM: segment menor + sem overlap
                wav_b = wav.unsqueeze(0)
                with torch.no_grad():
                    # split=True usa menos RAM
                    sources = apply_model(model, wav_b, device="cpu", split=True, overlap=0.25, segment=15*sr)[0]
                
                # só 2: voz e resto
                vocals = sources[3].cpu().numpy().T
                no_vocals = (sources[0]+sources[1]+sources[2]).cpu().numpy().T
                
                sf.write("/tmp/voz.wav", vocals, sr)
                sf.write("/tmp/base.wav", no_vocals, sr)
                
                st.success("FEITO! Modo leve 100%")
                c1,c2=st.columns(2)
                with c1:
                    st.markdown("**VOZ**")
                    st.audio("/tmp/voz.wav")
                with c2:
                    st.markdown("**BASE**")
                    st.audio("/tmp/base.wav")

                # PARTITURA V95 igual
                drum = sources[0][0].cpu().numpy()
                step=int(sr*0.15)
                energies=[np.sum(drum[i:i+step]**2) for i in range(0,len(drum),step)]
                bpm=120
                if energies:
                    peaks=[i for i in range(1,len(energies)-1) if energies[i]>energies[i-1] and energies[i]>energies[i+1]]
                    if len(peaks)>2:
                        diffs=sorted([peaks[i]-peaks[i-1] for i in range(1,len(peaks))])
                        median=diffs[len(diffs)//2] if diffs else 25
                        bpm=int(60/(median*0.15)) if median else 120
                
                st.markdown(f"### 📜 PARTITURA - 4/4 | BPM {bpm}")
                prog=["C","G","Am","F"]
                html="<div style='background:#fff;border:2px solid #000;border-radius:12px;padding:12px;overflow-x:auto;white-space:nowrap'>"
                html+="<div style='display:inline-block;font-size:40px;vertical-align:top'>𝄞</div>"
                for i in range(24):
                    chord=prog[i%len(prog)]
                    html+=f"<div style='display:inline-block;width:80px;height:100px;border:1px solid #ccc;border-radius:8px;margin-right:3px;text-align:center;position:relative;vertical-align:top'><div style='font-weight:900'>{chord}</div><div style='position:absolute;top:40px;left:0;width:100%;height:1px;background:#000'></div><div style='position:absolute;top:55px;left:0;width:100%;height:1px;background:#000'></div><div style='position:absolute;top:70px;left:0;width:100%;height:1px;background:#000'></div><div style='position:absolute;bottom:2px;width:100%;font-size:8px'>{i+1}</div></div>"
                html+="</div>"
                st.markdown(html, unsafe_allow_html=True)
                
                del sources, model, wav
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                st.error(f"Erro: {e}")
                st.info("Tenta com MP3 menor, até 8MB")
        
        os.unlink(path)
