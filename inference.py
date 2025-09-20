import torch, cv2
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image

# ---------- EDIT ----------
model_path = r"C:\Deep Learning\capstone_project\models\pneumonia_classifier.pth"
sample_img = r"C:\Deep Learning\capstone_project\sample_xray3.png"
# --------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

img = Image.open(sample_img).convert("L")
inp = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    prob = torch.softmax(model(inp), dim=1)[0].cpu().numpy()
    pred = int(prob.argmax())

label_map = ["NORMAL","PNEUMONIA"]
color = (0,255,0) if pred==0 else (0,0,255)
txt = f"{label_map[pred]} {prob[pred]:.2f}"

img_np = cv2.imread(sample_img)
h,w = img_np.shape[:2]
cv2.rectangle(img_np, (10,10), (w-10,h-10), color, 4)
cv2.putText(img_np, txt, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

plt.figure(figsize=(6,6)); plt.axis('off')
plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
plt.show()
