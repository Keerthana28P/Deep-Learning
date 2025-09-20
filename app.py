import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# --------------------
# Load your trained ResNet18 model
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)   # 2 classes: Normal, Pneumonia

model.load_state_dict(torch.load(
    r"C:\Deep Learning\capstone_project\models\pneumonia_classifier.pth",
    map_location=device
))
model.to(device)
model.eval()

# --------------------
# Transform
# --------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# --------------------
# Prediction + annotation
# --------------------
def predict_and_annotate(img: Image.Image):
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = "PNEUMONIA" if pred.item() == 1 else "NORMAL"
    conf_score = confidence.item() * 100

    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    color = (0,255,0) if label=="NORMAL" else (0,0,255)  # Green=Normal, Red=Pneumonia

    # Draw bounding box around full image
    cv2.rectangle(img_cv, (10,10), (w-10,h-10), color, 4)

    # Put label + confidence score
    text = f"{label}: {conf_score:.2f}%"
    cv2.putText(img_cv, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, color, 3, cv2.LINE_AA)

    result_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return result_img, label, conf_score

# --------------------
# Streamlit UI
# --------------------
st.title("Pneumonia Detection (ResNet18)")

uploaded_file = st.file_uploader("Upload a Chest X-Ray (.png/.jpg)", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    result_img, label, conf_score = predict_and_annotate(image)

    st.subheader(f"Prediction: {label} ({conf_score:.2f}%)")
    st.image(result_img, caption="Result with Bounding Box", use_container_width=True)