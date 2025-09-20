import os, random, shutil
import pydicom, numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ---------- EDIT ----------
csv_path   = r"C:\Deep Learning\capstone_project\data\raw\stage_2_train_labels.csv"
dcm_dir    = r"C:\Deep Learning\capstone_project\data\raw\stage_2_train_images"
out_root   = r"C:\Deep Learning\capstone_project\data\processed\balanced_dataset"
N_PER_CLASS = 200  # small quick subset
# --------------------------

os.makedirs(os.path.join(out_root, "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(out_root, "PNEUMONIA"), exist_ok=True)

df = pd.read_csv(csv_path)
df["label"] = (df["Target"] > 0).astype(int)

pos = df[df.label==1].sample(n=min(N_PER_CLASS, sum(df.label==1)), random_state=42)
neg = df[df.label==0].sample(n=min(N_PER_CLASS, sum(df.label==0)), random_state=42)
subset = pd.concat([pos, neg]).reset_index(drop=True)

def save_dcm_as_png(dcm_path, png_path):
    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(float)
    norm = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)
    Image.fromarray(norm).save(png_path)

for _, row in tqdm(subset.iterrows(), total=len(subset)):
    pid = row["patientId"]
    src = os.path.join(dcm_dir, pid + ".dcm")
    tgt = os.path.join(out_root, "PNEUMONIA" if row["label"]==1 else "NORMAL", pid+".png")
    save_dcm_as_png(src, tgt)

print("Done. Images saved in:", out_root)
