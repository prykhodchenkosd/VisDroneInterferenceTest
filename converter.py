import os
from glob import glob

# =========================
# CONFIG
# =========================

VISDRONE_ANN_DIR = "dataset/labels/valx"
VISDRONE_IMG_DIR = "dataset/images/val"

#YOLO_LABELS_DIR = "dataset/labels/train"
YOLO_LABELS_DIR = "dataset/labels/val2"

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
print(os.path.exists(VISDRONE_ANN_DIR))
print(os.path.exists(VISDRONE_IMG_DIR))
# =========================
# CLASS MAPPING
# =========================
# VisDrone classes:
#
# 0 ignored regions
# 1 pedestrian
# 2 people
# 3 bicycle
# 4 car
# 5 van
# 6 truck
# 7 tricycle
# 8 awning-tricycle
# 9 bus
# 10 motor
#
# We keep only:
# pedestrian + people -> person
# car
# truck
# bus

CLASS_MAP = {
    1: 0,  # pedestrian -> person
    2: 0,  # people -> person
    4: 1,  # car
    6: 2,  # truck
    9: 3   # bus
}

# =========================
# CONVERSION
# =========================

annotation_files = glob(os.path.join(VISDRONE_ANN_DIR, "*.txt"))

for ann_path in annotation_files:

    filename = os.path.basename(ann_path)

    image_name = filename.replace(".txt", ".jpg")

    image_path = os.path.join(VISDRONE_IMG_DIR, image_name)

    if not os.path.exists(image_path):
        continue

    # Read image size
    import cv2

    image = cv2.imread(image_path)

    if image is None:
        continue

    height, width = image.shape[:2]

    yolo_lines = []

    with open(ann_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split(",")

        if len(parts) < 6:
            continue

        x = int(parts[0])
        y = int(parts[1])
        w = int(parts[2])
        h = int(parts[3])

        category = int(parts[5])

        # Skip unwanted classes
        if category not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[category]

        # Convert to YOLO format
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height

        w_norm = w / width
        h_norm = h / height

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

        yolo_lines.append(yolo_line)

    # Save YOLO label
    output_path = os.path.join(YOLO_LABELS_DIR, filename)

    with open(output_path, "w") as f:
        f.write("\n".join(yolo_lines))

print("VisDrone -> YOLO conversion completed.")