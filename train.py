from ultralytics import YOLO
import multiprocessing
import torch

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        device=0,
        epochs=50,
        imgsz=640,
        batch=16,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()