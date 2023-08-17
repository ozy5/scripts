import os
import cv2
import numpy as np

images_path = "/home/umut/Desktop/IEEE_disaster_paper/datasets/NII_CU_MAPD_dataset/4-channel/images/thermal"



labels_path = "/home/umut/Desktop/IEEE_disaster_paper/datasets/NII_CU_MAPD_dataset/4-channel/labels"




for root, dirs, files in os.walk(images_path):
    for img_name in files:
        current_img_path = os.path.join(root, img_name)
        labels_name = os.path.splitext("/".join(current_img_path.split("/")[-2:]))[0] + ".txt"
        current_label_path = os.path.join(labels_path, labels_name)
        print(current_img_path)
        print(current_label_path)
        print("\n")

