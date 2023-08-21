import os
import shutil
import glob
import cv2
from imageio import imwrite
from gray_scale_to_rgb import gray_to_rgb_filter

SOURCE_DATASET_PATH = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset"

DEST_DATASET_PATH = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset_filtered_170_150"


for sub_folder_path in glob.glob(os.path.join(SOURCE_DATASET_PATH, "**", "**")):
    os.makedirs(os.path.join(DEST_DATASET_PATH, "/".join(sub_folder_path.split("/")[-2:])))

os.makedirs(os.path.join(DEST_DATASET_PATH, "train"), exist_ok=True)




data_yaml_path = os.path.join(SOURCE_DATASET_PATH, "data.yaml")
if os.path.exists(data_yaml_path):
    shutil.copy(data_yaml_path, os.path.join(DEST_DATASET_PATH, "data.yaml"))


paths = glob.glob(os.path.join(SOURCE_DATASET_PATH, "**", "**", "*"))

for path in paths:
    folder_name, images_or_labels, name = path.split("/")[-3:]
    save_name = os.path.join(DEST_DATASET_PATH, folder_name, images_or_labels, name)

    if images_or_labels == "images":
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


        bgr_filtered_img = gray_to_rgb_filter(img)


        rgb_filtered_img = bgr_filtered_img[:, :, ::-1]

        imwrite(save_name, rgb_filtered_img)
    elif images_or_labels == "labels":
        shutil.copy(path, save_name)















