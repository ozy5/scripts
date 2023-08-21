import os
import numpy as np
import cv2
from gray_scale_to_rgb import gray_to_rgb_filter
import time
from imageio import imwrite

SOURCE_PATH = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset/val/images"

#previous_was_135_100
DEST_PATH = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset_filtered/val/images"


#Filter to HIT_UAV_clean_final_filtered is 200 at blue, 160 at red channel

#Filter to HIT_UAV_clean_final_filtered is 200 at blue, 160 at red channel

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


