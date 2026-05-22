import cv2
import numpy as np


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)



def sharpen(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)



def preprocess_frame(image):
    image = apply_clahe(image)
    image = sharpen(image)

    return image