import os
import shutil
import pandas as pd

CSV_FILE = "data/patient_labels.csv"
SOURCE_DIR = "data/processed_slices"
DEST_DIR = "data/processed_slices_by_class"
os.makedirs(DEST_DIR, exist_ok=True)

# Read CSV
df = pd.read_csv(CSV_FILE)
df['PatientID'] = df['PatientID'].astype(str).str.strip().str.lower()
df['Diagnosis'] = df['Diagnosis'].astype(str).str.strip()

# Create folders based on CSV diagnosis values
for full_name in df['Diagnosis'].unique():
    if pd.notna(full_name):
        os.makedirs(os.path.join(DEST_DIR, full_name), exist_ok=True)

# Move images
for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".png"):
        patient_id = filename.split("_")[0].strip().lower()
        row = df[df['PatientID'] == patient_id]
        if row.empty:
            print(f"⚠️ No CSV entry for {patient_id}")
            continue
        diagnosis = row['Diagnosis'].values[0].strip()
        if pd.isna(diagnosis) or diagnosis == "":
            print(f"⚠️ Empty diagnosis for {patient_id}")
            continue
        dest_folder = os.path.join(DEST_DIR, diagnosis)
        shutil.move(os.path.join(SOURCE_DIR, filename), os.path.join(dest_folder, filename))
        print(f"✅ Moved {filename} to {diagnosis}")

print("✅ Images moved to class folders in", DEST_DIR)
