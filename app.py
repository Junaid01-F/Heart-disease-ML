import os
import time
import numpy as np
import tensorflow as tf
import cv2
import wikipedia

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
MODEL_PATH = "models/heart_cnn_final.h5"
UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

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

    return prediction, severity, preds.tolist(), class_index


# =========================
# Grad-CAM Heatmap
# =========================
def generate_gradcam(model, img_path, predicted_class):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    output_path = os.path.join(HEATMAP_FOLDER, "heatmap.jpg")
    cv2.imwrite(output_path, overlay)

    return "heatmap.jpg"


# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    severity = None
    filename = None
    preds = []
    heatmap_path = None
    info_text = ""

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

            prediction, severity, preds, class_index = predict_image(filepath)
            heatmap_path = generate_gradcam(model, filepath, class_index)

        else:
            return render_template("index.html", error="Please upload an image.")

    # Wikipedia Info
    if prediction:
        try:
            info_text = wikipedia.summary(
                prediction.replace("_", " "), sentences=4
            )
        except:
            info_text = "Wikipedia information not available."

    return render_template(
        "index.html",
        prediction=prediction,
        severity=severity,
        filename=filename,
        preds=preds,
        disease_classes=DISEASE_CLASSES,
        heatmap_path=heatmap_path,
        info_text=info_text,
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
