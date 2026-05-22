import cv2
import os
import numpy as np
from glob import glob
INPUT_DIR = "dataset/images/val"
OUTPUT_DIR = "dataset/images/val_degraded_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_smoke(image, intensity=0.5):
    h, w = image.shape[:2]

    smoke = np.full((h, w, 3), 255, dtype=np.uint8)

    alpha = intensity

    result = cv2.addWeighted(image, 1 - alpha, smoke, alpha, 0)

    return result


def add_dust(image, intensity=0.4):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)

    dusty = cv2.add(image, noise)

    dusty = cv2.GaussianBlur(dusty, (9, 9), 0)

    return cv2.addWeighted(image, 1 - intensity, dusty, intensity, 0)


def motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))

    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    kernel = kernel / kernel_size

    blurred = cv2.filter2D(image, -1, kernel)

    return blurred


def combined(image):
    image = add_smoke(image, 0.4)
    image = add_dust(image, 0.3)
    image = motion_blur(image, 17)

    return image


images = glob(os.path.join(INPUT_DIR, "*.jpg"))
print(f"Number of images: {len(images)}")

for img_path in images:
    img = cv2.imread(img_path)

    filename = os.path.basename(img_path)

    smoke_img = add_smoke(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"smoke_{filename}"), smoke_img)

    dust_img = add_dust(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"dust_{filename}"), dust_img)

    blur_img = motion_blur(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"blur_{filename}"), blur_img)

    combined_img = combined(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"combined_{filename}"), combined_img)

print("Degradation generation completed")