from ultralytics import YOLO

# Load YOLOv8m model (pretrained)
model = YOLO("yolov8m.pt")

# Train the model
model.train(
    data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",  # dataset path
    epochs=100,           # total epochs
    imgsz=640,            # image size
    batch=16,             # batch size (tune based on VRAM)
    device=0,             # GPU device (0 = RTX 4060)
    name="train_yolov8m_100ep",  # experiment name
    project=r"C:\Users\jeyar\wire_detection_project\runs\detect",  # output directory
    save_period=10,       # save checkpoint every 10 epochs
    patience=20,          # early stop if no improvement
    workers=8,            # data loading threads
    verbose=True          # show full logs
)

# Evaluate model performance after training
metrics = model.val()

