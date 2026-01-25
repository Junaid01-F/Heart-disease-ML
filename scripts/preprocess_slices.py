import os
import nibabel as nib
import numpy as np
from PIL import Image
import csv

# Paths
DATA_DIR = "data/trainingPublic/training"
LABEL_CSV = "data/patient_labels.csv"
OUTPUT_DIR = "data/processed_slices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read labels
patient_labels = {}
with open(LABEL_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        patient_labels[row["PatientID"]] = row["Diagnosis"]

# Function to normalize volume to 0-255
def normalize_to_uint8(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - mn) / (mx - mn)
    return (norm * 255).astype(np.uint8)

# Process each patient
patients = sorted(patient_labels.keys())
for p in patients:
    pdir = os.path.join(DATA_DIR, p)
    label = patient_labels[p]
    out_class_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_class_dir, exist_ok=True)

    # Find all .nii files in patient folder
    nii_files = [f for f in os.listdir(pdir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    for fn in nii_files:
        path = os.path.join(pdir, fn)
        try:
            img = nib.load(path)
            data = img.get_fdata()
            # If 4D, take first timepoint
            if data.ndim == 4:
                data = data[..., 0]
            # Take middle 3 slices along z-axis
            nz = data.shape[2]
            slice_indices = [nz//4, nz//2, 3*nz//4]
            for idx in slice_indices:
                slice2d = data[:, :, idx]
                slice2d = normalize_to_uint8(slice2d)
                im = Image.fromarray(slice2d)
                im = im.resize((128,128))
                outname = f"{p}_{fn.replace('.','_')}_slice{idx}.png"
                outpath = os.path.join(out_class_dir, outname)
                im.save(outpath)
        except Exception as e:
            print(f"ERROR {p}/{fn}: {e}")

print("✅ Preprocessing completed. Slices saved in", OUTPUT_DIR)
