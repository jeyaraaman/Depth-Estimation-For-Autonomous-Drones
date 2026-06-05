from ultralytics import YOLO

def main():
    model = YOLO(r"runs/obb/yolov8_obb_powerlines2/weights/best.pt")
    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_obb\data.yaml",
        epochs=50,
        imgsz=1024,
        batch=4,
        lr0=0.0008,
        freeze=10,
        project="runs/obb",
        name="final_merged_train"
    )

if __name__ == "__main__":
    main()
