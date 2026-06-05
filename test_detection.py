from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\runs\detect\train_wire_merged2\weights\best.pt")

# Run detection on your video or image folder
results = model.predict(
    source=0,  # or an image folder path
    show=True,     # shows detection live
    save=True,     # saves output in runs/predict/
    conf=0.4       # confidence threshold (0.25–0.5 works well)
)

print("✅ Detection complete! Check the 'runs/predict' folder for output files.")
