import streamlit as st
import tempfile

from utils.ml_model import classify_video
from utils.physics import calculate_physics

st.title("🎾 ML-Pendelanalyse")

video = st.file_uploader("MP4 hochladen", type=["mp4"])

length = st.number_input(
    "Pendellänge (m)",
    min_value=0.1,
    value=1.0
)

if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    st.video(tfile.name)

    with st.spinner("Analysiere Video..."):
        states, fps = classify_video(tfile.name)

        T, f, g = calculate_physics(states, fps, length)

    st.success("Analyse abgeschlossen")

    st.subheader("Ergebnisse")

    st.write(f"**Periodendauer:** {T:.2f} s")
    st.write(f"**Frequenz:** {f:.2f} Hz")
    st.write(f"**Erdbeschleunigung:** {g:.2f} m/s²")

    st.subheader("Erkannte Zustände")

    st.write(states[:30])
