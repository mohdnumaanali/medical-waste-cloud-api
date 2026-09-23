from ultralytics import YOLO

# 1. Load pre-trained classification model
model = YOLO('yolov8n-cls.pt')

# 2. Path to your unzipped dataset folder on D-drive
dataset_path = r"D:\PROJECTS\new pragatii\Medical Waste 4.0"

# 3. Start local training
results = model.train(
    data=dataset_path,
    epochs=10,
    imgsz=224,
    batch=16
)

print("✅ Model trained successfully! Best model saved in runs/classify/train/weights/best.pt")