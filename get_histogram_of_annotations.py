import os
import glob
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "/home/umut/Desktop/IEEE_disaster_paper/Training/nii_cu_dataset"

SAVE_PATH = "/home/umut/Desktop/IEEE_disaster_paper/Training/cropped_try"
os.makedirs(SAVE_PATH, exist_ok=True)

img_paths = glob.glob(os.path.join(DATASET_PATH, "*", "images", "*"))

label_paths = glob.glob(os.path.join(DATASET_PATH, "*", "labels", "*"))

sum_of_annotations_hist = np.zeros(256)
sum_of_image_hist = np.zeros(256)
annotation_count = 0
image_count = 0

for (img_path, label_path) in zip(sorted(img_paths, key=lambda x : os.path.basename(x)), sorted(label_paths, key=lambda x : os.path.basename(x))):
    img_name = os.path.splitext(img_path.split("/")[-1])[0]
    label_name = os.path.splitext(label_path.split("/")[-1])[0]


    if(img_name != label_name):
        print("IMAGE NAME DOES NOT MATCH WITH LABEL NAME")
        exit()

    img = ImageOps.grayscale(Image.open(img_path))
    image_width, image_height = img.size

    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for (i, line) in enumerate(lines):
        c, x, y, w, h = (float(x) for x in line.split(" "))

        x1 = int((x - w/2) * image_width)
        y1 = int((y - h/2) * image_height)
        x2 = int((x + w/2) * image_width)
        y2 = int((y + h/2) * image_height)

        im_cropped = img.crop((x1, y1, x2, y2))
        #im_cropped.save(os.path.join(SAVE_PATH, img_name + f"_{i}.png"))
        current_hist = np.array(im_cropped.histogram())
        sum_of_annotations_hist += current_hist
        annotation_count += 1
    sum_of_image_hist += np.array(img.histogram())
    image_count += 1


norm_sum_of_annotations_hist = sum_of_annotations_hist / np.sum(sum_of_annotations_hist)
norm_sum_of_image_hist = sum_of_image_hist / np.sum(sum_of_image_hist)

print(annotation_count, image_count)

plt.plot(norm_sum_of_annotations_hist, color="red")
plt.plot(norm_sum_of_image_hist, color="blue")
plt.show()
    
    