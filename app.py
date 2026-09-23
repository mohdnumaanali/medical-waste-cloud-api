from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="Smart Medical Waste Cloud API")

# Load trained YOLOv8 model weights
# Ensure best.pt is located at the root of your repository or update path accordingly
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Map medical waste categories to disposal bins
BIN_MAPPING = {
    "gauze": "YELLOW",
    "medical_cap": "YELLOW",
    "medical_filter": "YELLOW",
    "shoe_cover_pair": "YELLOW",
    "shoe_cover_single": "YELLOW",
    "glove_pair_latex": "RED",
    "glove_pair_nitrile": "RED",
    "glove_pair_surgery": "RED",
    "glove_single_latex": "RED",
    "glove_single_nitrile": "RED",
    "glove_single_surgery": "RED",
    "urine_bag": "RED",
    "medical_glasses": "BLUE",
    "test_tube": "BLUE"
}

CONFIDENCE_THRESHOLD = 0.75

@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "Smart Medical Waste Cloud Classification API is running!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read incoming image file from request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image payload received"}

    # Perform inference via YOLO model
    results = model(img, verbose=False)[0]
    top1_index = results.probs.top1
    class_name = results.names[top1_index]
    confidence = float(results.probs.top1conf.item())

    # Verify classification threshold
    if confidence >= CONFIDENCE_THRESHOLD and class_name in BIN_MAPPING:
        target_bin = BIN_MAPPING[class_name]
        is_medical = True
    else:
        target_bin = "GENERAL / UNKNOWN"
        is_medical = False

    return {
        "is_medical_waste": is_medical,
        "class": class_name,
        "confidence_percentage": round(confidence * 100, 2),
        "target_bin": target_bin
    }