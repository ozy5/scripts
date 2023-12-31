import os
from PIL import Image
import numpy as np
import glob
from ultralytics import YOLO
import cv2

SAVE_PATH_ROOT = "/home/umut/Desktop/THERMAL_DISASTER_VAL/test_for_thermal_disaster"

images_filtered_root_path = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset_filtered_200_175_hist_eq/test/images"

images_not_filtered_root_path = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset/test/images"

weights_filtered_path = "/home/umut/Desktop/THERMAL_DISASTER_VAL/HIT_UAV_and_NII_CU_dataset_filtered_200_175_hist_eq.pt"

weights_not_filtered_path = "/home/umut/Desktop/THERMAL_DISASTER_VAL/HIT_UAV_and_NII_CU_dataset.pt"

# THE NUMBER -3 IS PATH SPECIFIC
weights_filtered_name = weights_filtered_path.split("/")[-1]
weights_not_filtered_name = weights_not_filtered_path.split("/")[-1]


SAVE_PATH = os.path.join(SAVE_PATH_ROOT, weights_filtered_name + "_vs_" + weights_not_filtered_name)

os.makedirs(SAVE_PATH, exist_ok=True)



CONF_THRES = 0.2
IOU_THRES = 0.6
LINE_THICKNESS = 1


model_filtered = YOLO(weights_filtered_path)
model_not_filtered = YOLO(weights_not_filtered_path)



images_filtered= glob.glob(os.path.join(images_filtered_root_path, "*"))
images_not_filtered= glob.glob(os.path.join(images_not_filtered_root_path, "*"))

if(len(images_filtered) != len(images_not_filtered)):
    print("Not equal datasets by dataset sizes")
    exit()

for (filtered_img, not_filtered_img) in zip(sorted(images_filtered, key=lambda x : os.path.basename(x)), sorted(images_not_filtered, key=lambda x : os.path.basename(x))):
    filtered_img_name = filtered_img.split("/")[-1]
    not_filtered_img_name = not_filtered_img.split("/")[-1]

    if(filtered_img_name != not_filtered_img_name):
        print("Not equal datasets by image names")
        exit()

    #get the results
    filtered_img_results = model_filtered(filtered_img)[0]
    not_filtered_img_results = model_not_filtered(not_filtered_img)[0]

    #plot the bboxes
    annotated_filtered_img = filtered_img_results.plot(line_width = LINE_THICKNESS, conf_thres=CONF_THRES, iou_thres=IOU_THRES, show_labels=False, hide_conf=True)
    annotated_not_filtered_img = not_filtered_img_results.plot(line_width = LINE_THICKNESS, conf_thres=CONF_THRES, iou_thres=IOU_THRES, show_labels=False, hide_conf=True)

    # #add weights names as labels to upper-left corner of each annotated image
    annotated_filtered_img = cv2.putText(annotated_filtered_img, weights_filtered_name, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    annotated_not_filtered_img = cv2.putText(annotated_not_filtered_img, weights_not_filtered_name, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)



    #concatenate images horizontally
    concatenated_img = np.concatenate((annotated_filtered_img, annotated_not_filtered_img), axis=1)

    #turn BGR to RGB PIL image
    concatenated_img = Image.fromarray(cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB))

    #save the concatenated image
    concatenated_img.save(os.path.join(SAVE_PATH, filtered_img_name))

    
