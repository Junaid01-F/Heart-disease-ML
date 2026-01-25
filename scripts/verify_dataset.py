import os
import nibabel as nib
import numpy as np
import imageio

DATA_DIR = "data/trainingPublic/training"

OUT_PREVIEW_DIR = "data/preview_slices"
os.makedirs(OUT_PREVIEW_DIR, exist_ok=True)

def find_nii_files(d):
    return [f for f in os.listdir(d) if f.endswith(".nii") or f.endswith(".nii.gz")]

def normalize_to_uint8(arr):
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if np.isclose(mx, mn):
        return (np.zeros_like(arr) + 128).astype(np.uint8)
    norm = (arr - mn) / (mx - mn)
    return (norm * 255).astype(np.uint8)

patients = sorted([p for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))])
if not patients:
    print("No patient folders found in", DATA_DIR)
    raise SystemExit(1)

for p in patients:
    pdir = os.path.join(DATA_DIR, p)
    files = find_nii_files(pdir)
    print(f"\n=== {p} : {len(files)} .nii files ===")
    if not files:
        print("  (no .nii/.nii.gz files found — check folder contents)")
        continue

    for fn in files:
        path = os.path.join(pdir, fn)
        try:
            img = nib.load(path)
            data = img.get_fdata(dtype=np.float32)  
            print(f"  {fn} -> shape={data.shape}, dtype={data.dtype}")
            if data.ndim == 4:
                x,y,z,t = data.shape
                print(f"    detected 4D: spatial=({x},{y},{z}) timepoints={t}")
                vol = data[..., 0] 
            elif data.ndim == 3:
                x,y,z = data.shape
                print(f"    detected 3D: spatial=({x},{y},{z})")
                vol = data
            else:
                print("    Unsupported dims:", data.ndim)
                continue

        
            midz = vol.shape[2] // 2
            slice2d = vol[:, :, midz]
            img_u8 = normalize_to_uint8(slice2d)
            outname = f"{p}__{fn.replace('.','_')}_slice{midz}.png"
            outpath = os.path.join(OUT_PREVIEW_DIR, outname)
            imageio.imwrite(outpath, img_u8)
            print(f"    preview saved -> {outpath}")

        except Exception as e:
            print(f"  ERROR loading {fn}: {e}")
