import os

BASE_DIR = "data/processed_slices_by_class"

for cls in os.listdir(BASE_DIR):
    cls_path = os.path.join(BASE_DIR, cls)
    if os.path.isdir(cls_path):
        num_images = len([f for f in os.listdir(cls_path) if f.endswith(".png")])
        print(f"{cls}: {num_images} images")
