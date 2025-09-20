import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
import os

# ---------- EDIT ----------
data_dir = r"C:\Deep Learning\capstone_project\data\processed\balanced_dataset"
model_out = r"C:\Deep Learning\capstone_project\models\pneumonia_classifier.pth"
# --------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, num_epochs, lr = 16, 1, 1e-4

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    running = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {running/len(loader):.4f}")

# Classification Report
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in loader:
        outs = model(imgs.to(device))
        preds = outs.argmax(dim=1).cpu()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=dataset.classes))

os.makedirs(os.path.dirname(model_out), exist_ok=True)
torch.save(model.state_dict(), model_out)
print("Model saved:", model_out)
