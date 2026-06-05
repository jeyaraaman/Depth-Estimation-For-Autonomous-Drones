from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # small model
results = model.predict(source="0", show=False, device=0)  # webcam test on GPU
print("Detection complete ✅")