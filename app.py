import os
import time
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)

# =========================
# Paths & Config
# =========================
MODEL_PATH = "models/heart_cnn_final_optimized.h5"
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (128, 128)

DISEASE_CLASSES = [
    "Arrhythmogenic_right_ventricular_cardiomyopathy",
    "Dilated_cardiomyopathy",
    "Hypertrophic_cardiomyopathy",
    "Myocardial_infarction",
    "Normal",
    "Not_Heart"
]

# =========================
# Load Model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Make sure it exists in your repo."
    )

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")


# =========================
# Prediction Function
# =========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_index = int(np.argmax(preds))
    prediction = DISEASE_CLASSES[class_index]
    confidence = float(preds[class_index])

    # Severity Logic
    if prediction == "Normal":
        severity = "Healthy"
    elif prediction == "Not_Heart":
        severity = "N/A"
    else:
        if confidence < 0.4:
            severity = "Mild"
        elif confidence < 0.7:
            severity = "Moderate"
        else:
            severity = "Severe"

    return prediction, severity, preds.tolist()


# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    severity = None
    filename = None
    preds = []

    patient_name = "N/A"
    patient_age = "N/A"
    patient_gender = "N/A"
    patient_place = "N/A"

    if request.method == "POST":

        patient_name = request.form.get("patientName", "N/A")
        patient_age = request.form.get("patientAge", "N/A")
        patient_gender = request.form.get("patientGender", "N/A")
        patient_place = request.form.get("patientPlace", "N/A")

        file = request.files.get("file")

        if file and file.filename != "":
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            prediction, severity, preds = predict_image(filepath)

        else:
            return render_template("index.html", error="Please upload an image.")

    return render_template(
        "index.html",
        prediction=prediction,
        severity=severity,
        filename=filename,
        preds=preds,
        disease_classes=DISEASE_CLASSES,
        patientName=patient_name,
        patientAge=patient_age,
        patientGender=patient_gender,
        patientPlace=patient_place,
    )


# =========================
# Render PORT Config
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
