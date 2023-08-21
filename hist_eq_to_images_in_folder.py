import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from imageio import imwrite

SOURCE_PATH = "/home/umut/Desktop/thermal-disaster-dataset/others/selected_images_and_labels/images"
SAVE_PATH = "/home/umut/Desktop/thermal-disaster-dataset/others/selected_images_and_labels_hist_eq/images"

os.makedirs(SAVE_PATH, exist_ok=True)



for name in os.listdir(SOURCE_PATH):
    img_path = os.path.join(SOURCE_PATH, name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))
    img = clahe.apply(img)


    imwrite(os.path.join(SAVE_PATH, name), img)

