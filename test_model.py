import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Paths
MODEL_PATH = "models/heart_cnn_final.h5"
TEST_DIR = "data/processed_slices_by_class"

# Parameters
IMG_SIZE = (128, 128)

# Load model
model = load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# Map folders back to readable class names (optional)
class_names = sorted(os.listdir(TEST_DIR))  # this must match model training class order

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # scale

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    print(f"{os.path.basename(img_path)} → Predicted: {class_name} (Confidence: {prediction[0][class_index]:.2f})")

# Loop over all images in class folders
for folder in class_names:
    folder_path = os.path.join(TEST_DIR, folder)
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            if f.endswith(".png"):
                img_path = os.path.join(folder_path, f)
                predict_image(img_path)
