import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model Architecture
# ----------------------------
def load_model(model_path):

    model = models.resnet50(pretrained=False)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "resnet_finetuned_best.pth"
model = load_model(MODEL_PATH)


# ----------------------------
# Image Transform (Same as Training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------
# Prediction Function
# ----------------------------
def predict(image):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return pred.item(), confidence.item()


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("AI-Generated vs Real Image Detector")
st.write("Upload an image to check whether it is REAL or AI-generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, confidence = predict(image)

        if label == 0:
            result = "AI-Generated"
        else:
            result = "Real Photograph"

        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.4f}")
