from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    model.train(
        data="C:/Users/jeyar/wire_detection_project/yolo_gpu_env/datasets/merged_yolo/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name="train_merged_yolo"
    )
