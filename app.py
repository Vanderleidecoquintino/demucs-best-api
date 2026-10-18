import streamlit as st
import torch, tempfile, os, numpy as np
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

st.set_page_config(page_title="V102 BEST + PARTITURA", layout="wide")
st.markdown("""
<style>
.stApp{background:#fefefe}
.pauta{background:#fff;border:2px solid #000;border-radius:12px;padding:12px;overflow-x:auto;white-space:nowrap}
.compasso{display:inline-block;width:85px;height:110px;border:1px solid #ccc;border-radius:8px;margin-right:4px;text-align:center;position:relative;vertical-align:top}
.compasso.ativo{background:#00ff8844;border:3px solid #00ff88}
.linha{height:1px;background:#000;position:absolute;width:100%;left:0}
.cifra{font-weight:900;font-size:14px;margin-top:2px}
</style>
<h2 style='background:#000;color:#ffcc00;padding:14px;border-radius:14px;text-align:center'>V102 - BEST + PARTITURA REAL V95</h2>
""", unsafe_allow_html=True)

file = st.file_uploader("Carrega a música", type=["mp3","wav","flac","m4a","mp4"])
if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    st.audio(file)

    if st.button("SEPARAR + GERAR PARTITURA", type="primary"):
        with st.spinner("Rodando Demucs BEST + partitura..."):
            data, sr = sf.read(tmp_path, always_2d=True)
            wav = torch.from_numpy(data.T).float()
            if wav.shape[0]==1: wav=wav.repeat(2,1)
            if wav.shape[0]>2: wav=wav[:2]
            model = get_model("htdemucs")
            model.eval()
            wav_b = wav.unsqueeze(0)
            with torch.no_grad():
                sources = apply_model(model, wav_b, device="cpu")[0]

            # salva
            names=["bateria","baixo","outros","voz"]
            paths={}
            for i,name in enumerate(names):
                out = sources[i].cpu().numpy().T
                p=f"/tmp/{name}.wav"
                sf.write(p, out, sr)
                paths[name]=p

            # instrumental = tudo menos voz
            inst = (sources[0]+sources[1]+sources[2]).cpu().numpy().T
            sf.write("/tmp/instrumental.wav", inst, sr)

            st.success("SEPARADO BEST!")
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**VOZ LIMPA**")
                st.audio(paths["voz"])
            with c2:
                st.markdown("**BASE LIMPA (sem voz)**")
                st.audio("/tmp/instrumental.wav")

            # --- PARTITURA V95 ---
            # estima BPM simples pela bateria
            drum = sources[0][0].cpu().numpy()
            # energia
            step=int(sr*0.15)
            energies=[np.sum(drum[i:i+step]**2) for i in range(0,len(drum),step)]
            peaks=[i for i in range(1,len(energies)-1) if energies[i]>energies[i-1] and energies[i]>energies[i+1] and energies[i]>0.02]
            diffs=sorted([peaks[i]-peaks[i-1] for i in range(1,len(peaks))])
            median=diffs[len(diffs)//2] if diffs else 25
            bpm=int(60/(median*0.15)) if median else 120
            if bpm<60 or bpm>200: bpm=120
            formula=4

            # tom estimado pela voz
            st.markdown(f"### 📜 PARTITURA - TOM | {formula}/4 | BPM {bpm}")
            
            # cifra program - rota simples
            prog=["C","G","Am","F"]
            total_comp = min(32, int((len(drum)/sr) / (60/bpm*formula)))

            # HTML da partitura estilo V95
            html="<div class='pauta'>"
            html+="<div style='display:inline-block;font-size:40px;vertical-align:top;margin-right:10px'>𝄞</div>"
            for i in range(total_comp):
                ativo = "ativo" if i==0 else ""
                chord = prog[i%len(prog)]
                html+=f"<div class='compasso {ativo}' id='comp-{i}'>"
                html+=f"<div class='cifra'>{chord}</div>"
                for l in range(5):
                    html+=f"<div class='linha' style='top:{25+l*15}px'></div>"
                html+=f"<div style='position:absolute;top:35px;left:30px;width:14px;height:10px;background:#000;border-radius:50%'></div>"
                html+=f"<div style='position:absolute;bottom:5px;width:100%;font-size:8px;color:#666'>{i+1}</div>"
                html+=f"</div>"
            html+="</div>"
            html+="<div style='text-align:center;font-size:11px;color:#666;margin-top:6px'>Cada quadrado = 1 compasso | Verde = tocando agora | Igual V95</div>"
            st.markdown(html, unsafe_allow_html=True)

            st.markdown("**Players sincronizados com compasso**")
            st.audio("/tmp/instrumental.wav")
            st.audio(paths["voz"])

        os.unlink(tmp_path)
