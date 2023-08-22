import os
import glob
import utils.utils as utils
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from copy import deepcopy

CONF_THRESHOLD = 0.2

# IoU_threshold = 0.5
# IoU_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.9, 0.93, 0.96]
IoU_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.66, 0.69, 0.72, 0.75]
IoU_thresholds = list(np.linspace(0.1, 0.75, 20))
# FOR FILTERED:
#all images path
FILTERED_IMAGES_PATH = "/home/umut/Desktop/local_try_exp/HIT_UAV_and_NII_CU_dataset_filtered_200_175_hist_eq/test/images"

#all labels path (labels must be in YOLO format and has the same name with the corresponding image)
FILTERED_LABELS_PATH = "/home/umut/Desktop/local_try_exp/HIT_UAV_and_NII_CU_dataset_filtered_200_175_hist_eq/test/labels"

#YOLOv8 model path
FILTERED_MODEL_PATH = "/home/umut/Desktop/local_try_exp/filtered_HIT_UAV/filtered_new_200_175_hist_eq/weights/best.pt"



# FOR NOT_FILTERED:
#all images path
NOT_FILTERED_IMAGES_PATH = "/home/umut/Desktop/local_try_exp/HIT_UAV_and_NII_CU_dataset_filtered/test/images"

#all labels path (labels must be in YOLO format and has the same name with the corresponding image)
NOT_FILTERED_LABELS_PATH = "/home/umut/Desktop/local_try_exp/HIT_UAV_and_NII_CU_dataset_filtered/test/labels"

#YOLOv8 model path
NOT_FILTERED_MODEL_PATH = "/home/umut/Desktop/local_try_exp/filtered_HIT_UAV/filtered_new/weights/best.pt"


#load the model
model_filtered = YOLO(FILTERED_MODEL_PATH)
model_not_filtered = YOLO(NOT_FILTERED_MODEL_PATH)

filtered_model_recalls = []
filtered_model_precisions = []
filtered_model_f1_scores = []

not_filtered_model_recalls = []
not_filtered_model_precisions = []
not_filtered_model_f1_scores = []

#get the filtered results
results_filtered = model_filtered.predict(FILTERED_IMAGES_PATH, verbose=False)

# get prediction filtered results
predictions_xyxy_normalized_filtered = [(result.boxes.xyxyn.to("cpu")) for result in results_filtered]

#get label filtered paths
label_paths_filtered = [os.path.join(FILTERED_LABELS_PATH, (os.path.splitext(os.path.basename(result.path))[0] + ".txt")) for result in results_filtered]



#get the not filtered results
results_not_filtered = model_not_filtered.predict(NOT_FILTERED_IMAGES_PATH, verbose=False)

# get prediction not filtered results
predictions_xyxy_normalized_not_filtered = [(result.boxes.xyxyn.to("cpu")) for result in results_not_filtered]

#get label not filtered paths
label_paths_not_filtered = [os.path.join(NOT_FILTERED_LABELS_PATH, (os.path.splitext(os.path.basename(result.path))[0] + ".txt")) for result in results_not_filtered]

label_bboxes_filtered = [utils.get_xyxy_bboxes_from_YOLO_format_txt(label_path) for label_path in label_paths_filtered]
label_bboxes_not_filtered = [utils.get_xyxy_bboxes_from_YOLO_format_txt(label_path) for label_path in label_paths_not_filtered]


for IoU_threshold in IoU_thresholds:
    print(label_bboxes_filtered[0])

    # # FILTERED PART:

    #calculate the TP, FP and FN for filtered images
    TP_filtered, FP_filtered, FN_filtered = utils.calculate_TP_FP_FN_all_images(predictions_xyxy_normalized_filtered, label_bboxes_filtered, IoU_threshold=IoU_threshold)

    #calculate the recall and precision for filtered images
    recall_filtered, precision_filtered = utils.calculate_recall_and_precision_from_TP_FP_FN(TP_filtered, FP_filtered, FN_filtered)

    #calculate F1 score for filtered images
    f1_score_filtered = utils.get_F1_score_from_recall_and_precision(recall_filtered, precision_filtered)



    # # NOT FILTERED PART:




    #calculate the TP, FP and FN for not filtered images
    TP_not_filtered, FP_not_filtered, FN_not_filtered = utils.calculate_TP_FP_FN_all_images(predictions_xyxy_normalized_not_filtered, label_bboxes_not_filtered, IoU_threshold=IoU_threshold)

    #calculate the recall and precision for not filtered images
    recall_not_filtered, precision_not_filtered = utils.calculate_recall_and_precision_from_TP_FP_FN(TP_not_filtered, FP_not_filtered, FN_not_filtered)

    #calculate F1 score for not filtered images
    f1_score_not_filtered = utils.get_F1_score_from_recall_and_precision(recall_not_filtered, precision_not_filtered)


    # # STORE DATA FOR VISIUALIZATION PART

    filtered_model_recalls.append(recall_filtered)
    filtered_model_precisions.append(precision_filtered)
    filtered_model_f1_scores.append(f1_score_filtered)

    not_filtered_model_recalls.append(recall_not_filtered)
    not_filtered_model_precisions.append(precision_not_filtered)
    not_filtered_model_f1_scores.append(f1_score_not_filtered)



    # # PRINT PART

    print(f"For the IoU Threhold {IoU_threshold}:")
    print(f"\tFiltered:".ljust(15) + f"TP={TP_filtered}".ljust(10) + f"FP={FP_filtered}".ljust(10) + f"FN={FN_filtered}")
    print(f"\tNot_filtered: ".ljust(15) + f"TP={TP_not_filtered}".ljust(10) + f"FP={FP_not_filtered}".ljust(10) + f"FN={FN_not_filtered}")
    print("")
    print(f"\tFiltered:".ljust(15) + f"recall={round(recall_filtered, 3)}".ljust(14) + f"precision={round(precision_filtered, 3)}".ljust(18) + f"F1 score={round(f1_score_filtered, 3)}")
    print(f"\tNot filtered:".ljust(15) + f"recall={round(recall_not_filtered, 3)}".ljust(14) + f"precision={round(precision_not_filtered, 3)}".ljust(18) + f"F1 score={round(f1_score_not_filtered, 3)}")
    print(f"\n\n")


## VISIULATION PART

#plot 3 graphs as recall, precision and F1 score as subplots, side to side

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(IoU_thresholds, filtered_model_recalls, label="Filtered")
axs[0].plot(IoU_thresholds, not_filtered_model_recalls, label="Not Filtered")
axs[0].set_title("Recall")
axs[0].set_xlabel("IoU Threshold")
axs[0].set_ylabel("Recall")
axs[0].legend()

axs[1].plot(IoU_thresholds, filtered_model_precisions, label="Filtered")
axs[1].plot(IoU_thresholds, not_filtered_model_precisions, label="Not Filtered")
axs[1].set_title("Precision")
axs[1].set_xlabel("IoU Threshold")
axs[1].set_ylabel("Precision")
axs[1].legend()

axs[2].plot(IoU_thresholds, filtered_model_f1_scores, label="Filtered")
axs[2].plot(IoU_thresholds, not_filtered_model_f1_scores, label="Not Filtered")
axs[2].set_title("F1 Score")
axs[2].set_xlabel("IoU Threshold")
axs[2].set_ylabel("F1 Score")
axs[2].legend()

plt.suptitle("Filtered vs Not Filtered")
plt.show()



