import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="SNN Digit Recognizer", layout="wide")
st.title("🧠 SNN-Like Digit Recognizer (Rate-Coded MLP)")
st.caption("Upload a digit image → Model predicts the number")

# ------------------- MODEL DEFINITION -------------------
class SNN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ------------------- LOAD MODEL -------------------
device = "cpu"
model = SNN_MLP().to(device)

model_path = "model.pth"
if not os.path.exists(model_path):
    st.error("❌ model.pth not found. Upload a trained model file.")
    st.stop()

state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# ------------------- IMAGE TRANSFORM -------------------
transform = T.Compose([
    T.Grayscale(),
    T.Resize((28, 28)),
    T.ToTensor()
])

# ------------------- RATE CODING -------------------
def rate_code(img_tensor, steps=100):
    img_tensor = img_tensor.squeeze(0)
    img_tensor = img_tensor.repeat(steps, 1, 1)
    noise = torch.rand_like(img_tensor)
    spikes = (noise < img_tensor).float()
    return spikes.mean(dim=0).unsqueeze(0)

# ------------------- PREDICT -------------------
def predict(img):
    img = transform(img)

    # 100-step rate coding
    coded = rate_code(img, steps=100)

    with torch.no_grad():
        logits = model(coded.unsqueeze(0))
        pred = torch.argmax(logits, 1).item()

    return pred

# ------------------- UI -------------------
col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader("Upload Digit Image", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Input Digit", width=250)

        if st.button("🔍 Predict Digit"):
            with st.spinner("Processing with SNN-like rate coding..."):
                pred = predict(img)

            st.success(f"✅ Predicted Digit: **{pred}**")

with col2:
    st.subheader("📘 How It Works")
    st.markdown("""
    **This model uses:**
    - 100-step **Poisson rate coding**
    - SNN-style **surrogate gradient training**
    - A 3-layer MLP (ReLU forward-pass)
    """)

    st.info("""
    Make sure your `model.pth` is in the same folder.
    """)

