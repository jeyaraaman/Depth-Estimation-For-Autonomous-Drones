from ultralytics import YOLO

def main():
    # Load the last checkpoint
    model = YOLO(r"runs/detect/train_yolov8m_100ep/weights/last.pt")

    # Resume training from where it stopped (epoch 46)
    model.train(
        data=r"yolo_gpu_env/datasets/merged_yolo/data.yaml",
        epochs=100,          # target total epochs
        imgsz=640,           # image size
        batch=16,            # batch size (you can adjust)
        resume=True          # <-- this tells YOLO to continue training
    )

if __name__ == "__main__":
    main()
