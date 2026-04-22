"""
app.py — HAM10000 Skin Cancer Prediction API
Built specifically for the CNN trained in HAM10000.ipynb
Model is automatically downloaded from GitHub Releases on first startup.

Input:  POST /predict  — multipart image file (jpg/png)
Output: JSON with predicted class, confidence, and all probabilities
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import urllib.request

# ── Auto-download model from GitHub Releases if not present ───────────────────
MODEL_URL  = "https://github.com/JavariaRizwan/HAM10000_API/releases/download/v1.0/model.h5"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model.h5 from GitHub Releases...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ model.h5 downloaded successfully!")
else:
    print("✅ model.h5 already exists, skipping download.")

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HAM10000 Skin Lesion Classifier",
    description="Classifies dermatoscopic images into 7 skin lesion categories.",
    version="1.0.0"
)

# Allow any frontend (React, mobile, Postman) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model + config on startup ────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model("model.h5")
print("✅ Model loaded")

# Normalization stats saved from training (x_train_mean, x_train_std)
with open("norm_stats.json") as f:
    norm = json.load(f)
TRAIN_MEAN = norm["x_train_mean"]
TRAIN_STD  = norm["x_train_std"]

# Class names in the exact order LabelEncoder assigned them
with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)   # e.g. ['Actinic keratoses', 'Basal cell carcinoma', ...]

# Human-readable short-code → full name map
with open("label_map.json") as f:
    LABEL_MAP = json.load(f)     # e.g. {'nv': 'Melanocytic nevi', ...}

# Image dimensions — must match training (height=75, width=100)
IMG_HEIGHT = 75
IMG_WIDTH  = 100

print(f"✅ Classes: {CLASS_NAMES}")


# ── Helper ─────────────────────────────────────────────────────────────────────
def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """
    Replicates the exact preprocessing from the notebook:
      1. Resize to (100, 75)  — PIL uses (width, height)
      2. Convert to numpy array
      3. Normalize with training mean/std
      4. Reshape to (1, 75, 100, 3)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))          # PIL: (width, height)
    arr = np.asarray(img, dtype=np.float32)            # shape: (75, 100, 3)
    arr = (arr - TRAIN_MEAN) / TRAIN_STD               # same normalization as training
    arr = arr.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)     # add batch dimension
    return arr


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "HAM10000 Skin Lesion API is running.",
        "usage": "POST /predict with a skin lesion image file",
        "classes": CLASS_NAMES
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    try:
        img_bytes = await file.read()
        arr = preprocess_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {str(e)}")

    # Run inference
    preds = model.predict(arr)[0]           # shape: (7,)
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(np.max(preds))

    # Build full probability dict
    all_probs = {name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, preds)}

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "disclaimer": "This is a research tool, not a medical diagnosis. Consult a dermatologist."
    }


# ── Local dev entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)