import pandas as pd

# Load CSV
csv_file = "data/patient_labels.csv"
df = pd.read_csv(csv_file)

# Mapping abbreviations to full disease names
diagnosis_map = {
    "DCM": "Dilated_cardiomyopathy",
    "HCM": "Hypertrophic_cardiomyopathy",
    "MINF": "Myocardial_infarction",
    "NOR": "Normal",
    "RV": "Arrhythmogenic_right_ventricular_cardiomyopathy"
}

# Fill the Diagnosis column
df['Diagnosis'] = df['Group'].map(diagnosis_map)

# Save back
df.to_csv(csv_file, index=False)

print("✅ Diagnosis column updated with full disease names.")
