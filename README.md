# Heart Disease Prediction Using MRI Images

## Overview

This project presents a **heart disease prediction system** using **MRI images** and **deep learning (CNN)**. A web-based interface built with **Flask** allows users to 
upload a heart MRI image and receive:

* Predicted heart disease class
* Severity level (Mild / Moderate / Severe)
* Visual explanation using Grad-CAM
* Brief disease description via Wikipedia

The system is designed as a **decision-support tool** to demonstrate the application of explainable AI in medical imaging.

---

## Key Features

* CNN-based MRI image classification
* Severity estimation using confidence scores
* Grad-CAM heatmap for explainability
* Flask-based web application
* Clean and reproducible ML pipeline

---

## Technologies

* **Backend:** Python, TensorFlow/Keras, Flask
* **Frontend:** HTML, CSS, JavaScript
* **Libraries:** NumPy, OpenCV, Wikipedia API

---

## Model & Approach

A **Convolutional Neural Network (CNN)** is used due to its effectiveness in medical image analysis. CNNs automatically learn spatial and structural features from MRI images, 
making them suitable for detecting cardiac abnormalities.

**Predicted Classes:**

* Arrhythmogenic Right Ventricular Cardiomyopathy
* Dilated Cardiomyopathy
* Hypertrophic Cardiomyopathy
* Myocardial Infarction
* Normal
* Not Heart

**Severity Logic:**

* Normal → Healthy
* Not Heart → N/A
* Otherwise based on confidence score

---

## Explainability

Grad-CAM is used to highlight regions of the MRI image that influenced the model’s prediction, improving transparency and trust in the system.

---

## Project Structure

```
Heart-disease-ML/
├── app.py
├── src/
├── notebooks/
├── static/
├── templates/
├── models/        # Ignored (large files)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup & Usage

```bash
git clone https://github.com/your-username/Heart-disease-ML.git
cd Heart-disease-ML
pip install -r requirements.txt
python app.py
```

Access the application at `http://127.0.0.1:5000/`.

---

## Model Files

Trained model files (`.h5`, `.pkl`) are intentionally excluded from this repository due to size constraints and GitHub best practices.

---

## Limitations

* Requires high-quality MRI images
* Performance depends on dataset quality
* Intended for assistance, not clinical diagnosis

---

## Author
Junaid Pasha F.
