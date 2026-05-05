from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

# Modell laden
model = load_model("model/keras_model.h5", compile=False)

# Labels laden
class_names = open("model/labels.txt", "r").readlines()


def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)

    states = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV (BGR) -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame_rgb)

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        image_array = np.asarray(image)

        # Teachable Machine Normalisierung
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data, verbose=0)

        index = np.argmax(prediction)
        confidence = prediction[0][index]

        # Nur sichere Vorhersagen
        if confidence > 0.8:
            label = class_names[index].strip()
            states.append((frame_idx, label))

        frame_idx += 1

    cap.release()

    return states, fps
