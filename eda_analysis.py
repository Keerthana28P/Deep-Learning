import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch

# ----------------------------
# Paths
# ----------------------------
data_dir = r"C:\Deep Learning\capstone_project\data\processed\balanced_dataset"
output_file = r"C:\Deep Learning\capstone_project\data\processed\preprocessed_data.pt"

# ----------------------------
# EDA Part
# ----------------------------
classes = ["NORMAL", "PNEUMONIA"]

# Count images
img_counts = {cls: len(glob(os.path.join(data_dir, cls, "*.png"))) for cls in classes}
print("Image counts per class:", img_counts)

# Plot distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=list(img_counts.keys()), y=list(img_counts.values()), palette="viridis")
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.show()

# Show some sample images
plt.figure(figsize=(8, 4))
for i, cls in enumerate(classes):
    sample_img = glob(os.path.join(data_dir, cls, "*.png"))[0]
    img = Image.open(sample_img)
    plt.subplot(1, 2, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(cls)
    plt.axis("off")
plt.show()

# ----------------------------
# Preprocessing Part
# ----------------------------
def preprocess_image(path, img_size=(224, 224)):
    img = Image.open(path).convert("RGB")  # ensure 3 channels
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0      # normalize [0,1]
    return img_array

X, y = [], []

for idx, cls in enumerate(classes):
    files = glob(os.path.join(data_dir, cls, "*.png"))
    for f in files:
        X.append(preprocess_image(f))
        y.append(idx)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("Preprocessed dataset shape:", X.shape)  # (num_samples, 224, 224, 3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train).permute(0, 3, 1, 2)  # (N, C, H, W)
X_test_tensor = torch.tensor(X_test).permute(0, 3, 1, 2)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Save processed dataset
torch.save({
    "X_train": X_train_tensor,
    "X_test": X_test_tensor,
    "y_train": y_train_tensor,
    "y_test": y_test_tensor
}, output_file)

print(f"✅ Preprocessing complete! Data saved at: {output_file}")