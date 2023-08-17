import os
import numpy as np
import cv2
from gray_scale_to_rgb import gray_to_rgb_filter
import time
from imageio import imwrite

SOURCE_PATH = "DATASET_IMGS/nii_cu_dataset/val/images"


DEST_PATH = "DATASET_IMGS/nii_cu_dataset_filtered/val/images"

os.makedirs(DEST_PATH, exist_ok=True)


passed_time = 0

total_time = time.time()

img_count = len(os.listdir(SOURCE_PATH))

for img_name in os.listdir(SOURCE_PATH):

    img_path = os.path.join(SOURCE_PATH, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    time_start = time.time()

    bgr_filtered_img = gray_to_rgb_filter(img)

    passed_time += time.time() - time_start

    rgb_filtered_img = bgr_filtered_img[:, :, ::-1]

    imwrite(os.path.join(DEST_PATH, ( img_name)), rgb_filtered_img)

print(f"average time taken: {passed_time/img_count} at inferences")

print("total time taken: ", time.time() - total_time)


