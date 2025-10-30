# app.py
import streamlit as st
import torch
import os
from model import Model
from utils import ValidationDataset, predict, train_transforms

# Setting page configuration
st.set_page_config(page_title="DeepSafe", layout="wide", page_icon="🎭")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        text-align: center;
        font-size: 40px;
        color: #3E64FF;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 26px;
        margin-top: 20px;
    }
    .confidence {
        font-size: 24px;
        color: #28a745;
        font-weight: bold;
    }
    .footer {
        margin-top: 30px;
        text-align: center;
        color: #aaaaaa;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">🤍 Deepsafe : DeepFake Video Detector</div>', unsafe_allow_html=True)

device = torch.device('cpu')

@st.cache_resource
def load_model():
    model = Model(num_classes=2).to(device)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_89_acc_40_frames_final_data.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# Upload video
st.subheader("Upload a Video File (.mp4)")
uploaded_file = st.file_uploader("", type=["mp4"])

if uploaded_file is not None:
    # File size limit: 100MB (in bytes)
    max_file_size = 100 * 1024 * 1024

    if uploaded_file.size > max_file_size:
        st.error("❌ The uploaded file exceeds the 200MB size limit. Please upload a smaller video.")
    else:
        # Save video locally
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        dataset = ValidationDataset("temp_video.mp4", sequence_length=20, transform=train_transforms)
        video = dataset[0]

        with st.spinner('🔍 Analyzing the video...'):
            prediction, confidence, heatmap_img = predict(model, video)

        label = "REAL" if prediction == 1 else "FAKE"
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔹 Uploaded Video Preview")
            st.video("temp_video.mp4")
        
        with col2:
            st.subheader("🔸 Heatmap Visualization")
            st.image(heatmap_img, caption="Prediction Heatmap", use_container_width=True)

        # Centered final result
        st.markdown(
            f"<div style='text-align: center; margin-top: 40px;'><h2>📝 Prediction: <span style='color:#3E64FF;'>{label}</span></h2>"
            f"<h3 class='confidence'>Confidence: {confidence:.2f}%</h3></div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("<div class='footer'></div>", unsafe_allow_html=True)
