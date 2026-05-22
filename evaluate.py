"""
from ultralytics import YOLO
import pandas as pd
from ultralytics import YOLO
import multiprocessing
import torch

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    model = YOLO("runs/detect/train-9/weights/best.pt")
    results = {}
    sets = {
        "clean": "dataset/images/val",
        "smoke": "dataset/images/val_degraded_dataset/smoke",
        "dust": "dataset/images/val_degraded_dataset/dust",
        "blur": "dataset/images/val_degraded_dataset/blur",
        "combined": "dataset/images/val_degraded_dataset/combined"
    }
    for name, path in sets.items():
        metrics = model.val(data="dataset.yaml", split="val")
        results[name] = {
            "mAP50": metrics.box.map50,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr
        }
    df = pd.DataFrame(results).T
    print(df)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/results.csv")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
"""

from ultralytics import YOLO
import pandas as pd
import multiprocessing
import torch
import yaml
import os
import glob


def main():

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    for f in glob.glob("dataset/*.cache"):
        os.remove(f)
    model = YOLO("runs/detect/train-9/weights/best.pt")

    results = {}

    sets = {
        "clean": "images/val",
        "smoke": "images/val_degraded_dataset/smoke",
        "dust": "images/val_degraded_dataset/dust",
        "blur": "images/val_degraded_dataset/blur",
        "combined": "images/val_degraded_dataset/combined"
    }

    os.makedirs("temp_yaml", exist_ok=True)

    for name, val_path in sets.items():

        temp_data = {
            "path": "dataset",
            "train": "images/train",
            "val": val_path,
            "names": {
                0: "person",
                1: "car",
                2: "truck",
                3: "bus"
            }
        }

        yaml_path = f"temp_yaml/{name}.yaml"

        with open(yaml_path, "w") as f:
            yaml.dump(temp_data, f)

        print(f"Testing: {name}")
        print(temp_data)

        metrics = model.val(
            data=yaml_path,
            split="val"
        )

        results[name] = {
            "mAP50": metrics.box.map50,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr
        }

    df = pd.DataFrame(results).T

    print(df)

    os.makedirs("outputs", exist_ok=True)

    df.to_csv("outputs/results.csv")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()