# 🛡️ DeepSafe - DeepFake Detection Web App

DeepSafe is a Streamlit-based web application designed to detect whether a video is real or fake (DeepFake). It uses a trained deep learning model built with PyTorch that analyzes facial features and temporal patterns in videos.

## 🚀 Features

- 🎥 Upload any video clip.
- ⚙️ Model processes and analyzes frames.
- 🔍 Detects whether the video is Real or Fake.
- 🌡️ Displays prediction confidence and a heatmap.
- 🖥️ Runs completely on CPU or GPU (if available).

## 📦 Project Structure

```
DEEPSAFE/
│
├── app.py                         # Streamlit web app
├── model/
│   └── final_model.pt            # Trained model weights
├── utils/
│   ├── dataset.py                # Frame extraction logic
│   └── transforms.py            # Preprocessing & transforms
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
```

## 🧠 Model Info

The model is a hybrid architecture based on `ResNeXt50` and `LSTM` for video-level classification. It was trained using the DFDC dataset.

## 📥 Download Model Weights

Download the trained model (`model_89_acc_40_frames_final_data.pt`) from Google Drive:

🔗 [Download Weights](https://drive.google.com/file/d/1SynPVYcLp64T5KgitvVFBEJKMO86KW39/view?usp=sharing)

Place it in the following directory:

```
DEEPSAFE/model/final_model.pt
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/dishnat-hash-lab/DEEPSAFE.git
cd DEEPSAFE
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv deepfake-env
source deepfake-env/bin/activate  # On Windows: deepfake-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 💡 Notes

- The model runs on CPU if GPU is not available.
- Use videos of faces only for better accuracy.
- It selects a sequence of frames from the uploaded video and predicts based on temporal information.

---

## 🙌 Acknowledgements

- DFDC Dataset by Facebook AI
- Streamlit for UI
- PyTorch for model development

---
