from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()

    # Load the trained model as pretrained weights (not resume)
    model = YOLO(r"runs/detect/train_yolov8m_100ep/weights/last.pt")

    # Fine-tune starting from those weights
    model.train(
        data=r"C:\Users\jeyar\wire_detection_project\yolo_gpu_env\datasets\merged_yolo\data.yaml",
        epochs=150,                  # total fine-tuning epochs (this will train for 150 more epochs)
        imgsz=960,
        batch=2,
        device=0,
        amp=True,
        lr0=0.0005,
        optimizer="SGD",
        augment=True,
        workers=2,
        name="train_yolov8m_finetune_from100ep",
        resume=False                 # MUST stay False
    )

if __name__ == "__main__":
    main()
