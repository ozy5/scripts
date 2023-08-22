import os
import shutil
import glob
import cv2
from imageio import imwrite
from gray_scale_to_rgb import gray_to_rgb_filter, gray_to_rgb_with_clahe
import time


ROOT_SOURCE_PATH = "/home/umut/Desktop/newest_thermal_disaster_datasets/not_filtered_random_datasets"

DEST_PATH = "/home/umut/Desktop/newest_thermal_disaster_datasets/filtered_random_datasets"

os.makedirs(DEST_PATH, exist_ok=True)

for (index, dataset_name) in enumerate(sorted(os.listdir(ROOT_SOURCE_PATH))):
    time_start = time.time()

    current_new_dataset_name = dataset_name.replace("not_", "")

    current_dataset_source_path = os.path.join(ROOT_SOURCE_PATH, dataset_name)

    current_dataset_dest_path = os.path.join(DEST_PATH, current_new_dataset_name)


    for sub_folder_path in glob.glob(os.path.join(current_dataset_source_path, "**", "**")):
        os.makedirs(os.path.join(current_dataset_dest_path, "/".join(sub_folder_path.split("/")[-2:])), exist_ok=True)

    os.makedirs(os.path.join(current_dataset_dest_path, "train"), exist_ok=True)




    # data_yaml_path = os.path.join(current_dataset_source_path, "data.yaml")
    # if os.path.exists(data_yaml_path):
    #     shutil.copy(data_yaml_path, os.path.join(current_dataset_dest_path, "data.yaml"))


    paths = glob.glob(os.path.join(current_dataset_source_path, "**", "**", "*"))

    for path in paths:
        folder_name, images_or_labels, name = path.split("/")[-3:]
        save_name = os.path.join(current_dataset_dest_path, folder_name, images_or_labels, name)

        if images_or_labels == "images":
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


            bgr_filtered_img = gray_to_rgb_with_clahe(img)


            # save the bgr image converting to rgb
            imwrite(save_name, bgr_filtered_img[:, :, ::-1]) 
        elif images_or_labels == "labels":
            shutil.copy(path, save_name)
    
    print(f"For the dataset {dataset_name}, it took {round(time.time() - time_start, 2)} seconds to finish.")
