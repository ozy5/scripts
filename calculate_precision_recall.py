import os
import glob
import utils.utils as utils
import torch
from ultralytics import YOLO

CONF_THRESHOLD = 0.2

# IoU_threshold = 0.5
IoU_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# # # FOR FILTERED:
# # #all images path
# # IMAGES_PATH = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset_filtered_200_175_hist_eq/test/images"

# # #all labels path (labels must be in YOLO format and has the same name with the corresponding image)
# # LABELS_PATH = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset_filtered_200_175_hist_eq/test/labels"

# # #YOLOv8 model path
# # MODEL_PATH = "/home/umut/Desktop/THERMAL_DISASTER_VAL/HIT_UAV_and_NII_CU_dataset_filtered_200_175_hist_eq.pt"



# FOR NOT_FILTERED:
#all images path
IMAGES_PATH = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset/test/images"

#all labels path (labels must be in YOLO format and has the same name with the corresponding image)
LABELS_PATH = "/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset/test/labels"

#YOLOv8 model path
MODEL_PATH = "/home/umut/Desktop/THERMAL_DISASTER_VAL/HIT_UAV_and_NII_CU_dataset.pt"



#load the model
model = YOLO(MODEL_PATH)


for IoU_threshold in IoU_thresholds:
    #get the results
    results = model.predict(IMAGES_PATH, verbose=False)

    # get prediction results
    predictions_xyxy_normalized= [(result.boxes.xyxyn.to("cpu")) for result in results]

    #get label paths
    label_paths = [os.path.join(LABELS_PATH, (os.path.splitext(os.path.basename(result.path))[0] + ".txt")) for result in results]

    TP, FP, FN = utils.calculate_TP_FP_FN_all_images(predictions_xyxy_normalized, label_paths, IoU_threshold=IoU_threshold)

    recall, precision = utils.calculate_recall_and_precision_from_TP_FP_FN(TP, FP, FN)


    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"When the IoU threshold is:{IoU_threshold}\nrecall: {round(recall, 4)}, precision: {round(precision, 4)}")











