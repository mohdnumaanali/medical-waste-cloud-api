import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load trained classification model weights
model = YOLO('runs/classify/train-3/weights/best.pt')

# 2. Strict List of Trained Medical Waste Classes
MEDICAL_CLASSES = {
    "gauze": "Yellow Bin (Infectious Cotton Waste)",
    "glove_pair_latex": "Red Bin (Contaminated Gloves)",
    "glove_pair_nitrile": "Red Bin (Contaminated Gloves)",
    "glove_pair_surgery": "Red Bin (Contaminated Gloves)",
    "glove_single_latex": "Red Bin (Contaminated Gloves)",
    "glove_single_nitrile": "Red Bin (Contaminated Gloves)",
    "glove_single_surgery": "Red Bin (Contaminated Gloves)",
    "medical_cap": "Yellow Bin (Infectious Fabric)",
    "medical_filter": "Yellow Bin (Infectious Filter)",
    "medical_glasses": "Blue Bin (Recyclable Plastic/Goggles)",
    "shoe_cover_pair": "Yellow Bin (Infectious Covers)",
    "shoe_cover_single": "Yellow Bin (Infectious Covers)",
    "test_tube": "Blue Bin (Glassware / Lab Tube)",
    "urine_bag": "Red Bin (Contaminated Fluid Bag)"
}

# Require high confidence to accept an item as genuine medical waste
CONFIDENCE_THRESHOLD = 0.75  
MIN_MOTION_AREA = 1800      

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

last_box = None
last_label = "Waiting for Motion..."
last_bin = "Bin: None"
last_color = (128, 128, 128)

print("Medical Waste Motion Tracker Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Calculate motion difference between frames
    frame_delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    prev_gray = gray

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_MOTION_AREA]

        if valid_contours:
            # Get dynamic bounding box of moving object
            x_min = min([cv2.boundingRect(c)[0] for c in valid_contours])
            y_min = min([cv2.boundingRect(c)[1] for c in valid_contours])
            x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours])
            y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours])

            h_img, w_img, _ = frame.shape
            pad = 20
            x1, y1 = max(0, x_min - pad), max(0, y_min - pad)
            x2, y2 = min(w_img, x_max + pad), min(h_img, y_max + pad)

            last_box = (x1, y1, x2, y2)
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                results = model(roi, verbose=False)[0]
                top1_index = results.probs.top1
                class_name = results.names[top1_index]
                confidence = results.probs.top1conf.item()

                # Verify if predicted item is medical waste and passes confidence threshold
                if class_name in MEDICAL_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
                    destination = MEDICAL_CLASSES[class_name]
                    last_label = f"Medical Waste: {class_name} ({confidence * 100:.1f}%)"
                    last_bin = f"Bin: {destination}"
                    last_color = (0, 255, 0)  # Green box for medical waste
                else:
                    last_label = "NOT MEDICAL WASTE (Non-Medical Object)"
                    last_bin = "Bin: General Waste / Discard"
                    last_color = (0, 0, 255)  # Red box for non-medical objects

    # Draw moving box around object
    if last_box is not None:
        bx1, by1, bx2, by2 = last_box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), last_color, 3)

    # Display status on screen
    cv2.putText(frame, last_label, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)
    cv2.putText(frame, last_bin, (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

    cv2.imshow("Smart Medical Waste Hardware Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()