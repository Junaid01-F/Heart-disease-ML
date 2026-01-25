import os
import csv

DATA_DIR = "data/trainingPublic/training"
OUT_CSV = "data/patient_labels.csv"

patients = sorted([p for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))])

rows = []
for p in patients:
    pdir = os.path.join(DATA_DIR, p)
    info_file = os.path.join(pdir, "Info.cfg")  # sometimes "info.cfg" or "Info.txt"
    if not os.path.exists(info_file):
        info_file = os.path.join(pdir, "info.cfg")
    if not os.path.exists(info_file):
        print(f"⚠️ No info file for {p}")
        continue

    diagnosis = None
    group = None
    with open(info_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("group"):
                group = line.split(":")[-1].strip()
            if line.lower().startswith("diagnosis"):
                diagnosis = line.split(":")[-1].strip()

    rows.append([p, group, diagnosis])

# Save to CSV
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PatientID", "Group", "Diagnosis"])
    writer.writerows(rows)

print(f"✅ Saved labels for {len(rows)} patients to {OUT_CSV}")
