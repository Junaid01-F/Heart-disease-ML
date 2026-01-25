import os
import time
import numpy as np
import tensorflow as tf
import cv2
import wikipedia

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

MODEL_PATH = "models/heart_cnn_final.h5"
# Load model once
model = load_model(MODEL_PATH)
model.build((None, 128, 128, 3))
dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
_ = model(dummy_input)

print("✅ Model loaded successfully.")

def generate_gradcam(model, img_path, predicted_class):
    import tensorflow as tf
    import numpy as np
    import cv2
    import os

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if not last_conv_layer_name:
        print("[ERROR] No convolutional layer found in model.")
        return None

    last_conv_layer = model.get_layer(last_conv_layer_name)

    _ = model.predict(img_array)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        print("[WARNING] Gradient is None, returning blank heatmap.")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    output_path = os.path.join('static', 'heatmap.jpg')
    cv2.imwrite(output_path, overlay)

    return output_path

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

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_index = int(np.argmax(preds))
    prediction = DISEASE_CLASSES[class_index]
    confidence = float(preds[class_index])

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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    severity = None
    filename = None
    preds = []
    heatmap_path = None
    patient_name = "N/A"
    patient_age = "N/A"
    patient_gender = "N/A"
    patient_place = "N/A"  # New Variable

    if request.method == "POST":
        patient_name = request.form.get("patientName", "N/A")
        patient_age = request.form.get("patientAge", "N/A")
        patient_gender = request.form.get("patientGender", "N/A")
        patient_place = request.form.get("patientPlace", "N/A") # Capture Place

        file = request.files.get("file")
        if file and file.filename != "":
            timestamp = int(time.time())
            filename = f"{os.path.splitext(file.filename)[0]}_{timestamp}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            prediction, severity, preds = predict_image(filepath)
            predicted_class = int(np.argmax(preds))
     
            heatmap_path = generate_gradcam(model, filepath, predicted_class)
            if heatmap_path and heatmap_path.startswith("static/"):
                heatmap_path = heatmap_path[len("static/"):]
        else:
            return render_template(
                "index.html",
                error="Please upload an image.",
                patientName=patient_name,
                patientAge=patient_age,
                patientGender=patient_gender,
                patientPlace=patient_place,
            )

    info_text = ""
    if prediction:
        try:
            info_text = wikipedia.summary(prediction.replace("_", " "), sentences=4)
        except wikipedia.exceptions.DisambiguationError as e:
            info_text = f"Multiple results found for {prediction}. Please check Wikipedia directly."
        except wikipedia.exceptions.PageError:
            info_text = f"No Wikipedia page found for {prediction}."
        except Exception as e:
            info_text = f"Could not fetch Wikipedia info: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        severity=severity,
        filename=filename,
        patientName=patient_name,
        patientAge=patient_age,
        patientGender=patient_gender,
        patientPlace=patient_place, # Pass Place to template
        preds=preds,
        disease_classes=DISEASE_CLASSES,
        heatmap_path=heatmap_path,
        info_text=info_text
    )


if __name__ == "__main__":
    app.run(debug=True)