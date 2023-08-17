import os
from PIL import Image
import numpy as np

images_path = "/home/umut/Desktop/IEEE_disaster_paper/datasets/NII_CU_MAPD_dataset/4-channel/images/thermal"



labels_path = "/home/umut/Desktop/IEEE_disaster_paper/datasets/NII_CU_MAPD_dataset/4-channel/labels"

labels_dest_path =  "/home/umut/Desktop/IEEE_disaster_paper/datasets/NII_CU_MAPD_dataset/YOLO_DATASET/nii_cu_dataset"

os.makedirs(os.path.join(labels_dest_path, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(labels_dest_path, "val", "labels"), exist_ok=True)



for folder in os.listdir(images_path):
    for img_name in os.listdir(os.path.join(images_path, folder)):

        current_img_path = os.path.join(images_path, folder, img_name)

        current_img_rgb = Image.open("rgb".join(current_img_path.split("thermal")))
        
        w, h=  current_img_rgb.size
        

        label_name = ".".join(img_name.split(".")[:-1]) + ".txt"
        current_label_path = os.path.join(labels_path, folder, label_name)

        print(current_img_path)


        with open(current_label_path, "r") as f:
            lines = f.readlines()
        
        with open(os.path.join(labels_dest_path, folder, "labels", label_name), "w") as f:

            for line in lines:
                x1, y1, x2, y2, visibility, occluded, bad = line.split("\t")
                x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
                visibility, occluded, bad = int(visibility), int(occluded), int(bad)

                if(visibility == 2):
                    continue

                f.write(f"0 {((x1) + (x2)) / (2*w)} {((y1) + (y2)) / (2*h)} {((x2) - (x1)) / w} {((y2) - (y1)) / h}\n")
            
                