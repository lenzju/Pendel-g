import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tempfile
import time

# --- SETUP & KONFIGURATION ---
st.set_page_config(page_title="KI-Pendel-Analyse", layout="wide")
st.title("🎾 Pendel-Analyse: KI im Physikunterricht")

# Sidebar für Parameter
st.sidebar.header("Physikalische Parameter")
l_cm = st.sidebar.number_input("Pendellänge (in cm)", min_value=1.0, value=50.0)
l_m = l_cm / 100.0  # Umrechnung in Meter

# --- KI MODELLE LADEN ---
# Säule A: MediaPipe (Vortrainiert)
mp_hands = mp.solutions.hands # Alternativ Objectron oder Hands als Proxy für Tracking
# Da MediaPipe kein direktes "Tennisball"-Modell hat, nutzen wir die Object-Detection Logik 
# oder einfach Kontur-Tracking als Ergänzung zu Säule A.

# Säule B: Teachable Machine (Selbsttrainiert)
@st.cache_resource
def load_custom_model():
    try:
        model = load_model("model/keras_model.h5", compile=False)
        labels = open("model/labels.txt", "r").readlines()
        return model, labels
    except:
        return None, None

model_b, labels_b = load_custom_model()

# --- DATEI-UPLOAD ---
video_file = st.file_uploader("Lade ein MP4-Video deines Pendels hoch", type=["mp4", "mov"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    x_positions = []
    timestamps = []
    classifications = []
    
    st.info("Video wird analysiert... Bitte warten.")
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- VERARBEITUNGSSCHLEIFE ---
    curr_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Bildvorbereitung
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame_rgb, (224, 224)) # Für Teachable Machine
        
        # Säule A: Tracking (Hier via Farb-Kontur-Tracking als "Ersatz" für Objekterkennung)
        # Tennisball ist gelb -> HSV Farbraum
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                x_positions.append(cX)
                timestamps.append(curr_frame / fps)

        # Säule B: Klassifizierung (Teachable Machine)
        if model_b:
            normalized_img = (img_resized.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_img
            prediction = model_b.predict(data, verbose=0)
            class_idx = np.argmax(prediction)
            classifications.append(labels_b[class_idx].strip())

        curr_frame += 1
        if curr_frame % 10 == 0:
            progress_bar.progress(curr_frame / frame_count)

    cap.release()

    # --- AUSWERTUNG ---
    st.subheader("Ergebnisse der Analyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Säule A: Bewegungs-Tracking (Position-Zeit)**")
        if x_positions:
            df = pd.DataFrame({"Zeit (s)": timestamps, "X-Position": x_positions})
            st.line_chart(df.set_index("Zeit (s)"))
            
            # Physik-Berechnung: Periodendauer T
            # Einfache Schätzung über Maxima der X-Position
            peaks = (np.diff(np.sign(np.diff(x_positions))) < 0).nonzero()[0] + 1
            if len(peaks) > 1:
                T = (timestamps[peaks[-1]] - timestamps[peaks[0]]) / (len(peaks) - 1) * 2
                f = 1/T
                # g = (4 * pi^2 * l) / T^2
                g_calc = (4 * np.pi**2 * l_m) / (T**2)
                
                st.metric("Periodendauer (T)", f"{T:.2f} s")
                st.metric("Frequenz (f)", f"{f:.2f} Hz")
                st.metric("Berechnetes g", f"{g_calc:.2f} m/s²")
            else:
                st.warning("Nicht genug Schwingungen erkannt.")

    with col2:
        st.write("**Säule B: Zustands-Klassifizierung (KI)**")
        if classifications:
            class_df = pd.Series(classifications).value_counts()
            st.bar_chart(class_df)
            st.write("Häufigste Zustände:", class_df)

    # --- REFLEXION FÜR DOKUMENTATION ---
    st.divider()
    st.subheader("Kritische Reflexion (für KEL-Dokumentation)")
    st.write("""
    *   **Säule A (Tracking):** Liefert präzise Koordinaten für physikalische Berechnungen, ist aber anfällig für Lichtveränderungen (Gelb-Filter).
    *   **Säule B (Klassifizierung):** Erkennt den 'Zustand' (Links/Rechts), kann aber ohne Zeitstempel-Analyse keine exakten physikalischen Konstanten wie *g* berechnen.
    """)
