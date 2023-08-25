import os
import subprocess
import argparse

DATASETS_PATH = "/home/umut/Desktop/newest_thermal_disaster_datasets/hist_eq_filtered_variations/200_175/datasets"


ap = argparse.ArgumentParser()

ap.add_argument("-d", "--datasets", required=True, help="datasets_path")
ap.add_argument("-n", "--name", required=True,help="training_name (ex: 200_175)")
ap.add_argument("-e", "--epochs", required=True,help="epoch_number (ex: 100, 150, 200)")
ap.add_argument("-b", "--batch", required=True,help="batch_number (ex: 16, 32, 64)")
ap.add_argument("-c", "--comet", required=True, help="comet_project_name")

args = vars(ap.parse_args())

DATASETS_PATH = str(args["datasets"])
EPOCH_NUMBER = int(args["epochs"])
BATCH_NUMBER = int(args["batch"])
EXP_NAME = str(args["name"])
COMET_PROJECT_NAME = str(args["comet"])

print(f"Epoch number: {EPOCH_NUMBER}, Batch number: {BATCH_NUMBER}, Experiment name: {EXP_NAME}, Comet project name: {COMET_PROJECT_NAME}")




COMET_API_EXPORT = "export COMET_API_KEY=CtTXsL5EVp349Idk2fttJxhTL"
COMET_PROJECT_NAME_EXPORT = "export COMET_PROJECT_NAME=" + COMET_PROJECT_NAME

subprocess.run(COMET_API_EXPORT, shell=True)
subprocess.run(COMET_PROJECT_NAME_EXPORT, shell=True)





dataset_names = sorted(os.listdir(DATASETS_PATH))

dataset_count = len(dataset_names)

print(f"Starting to training {dataset_count} datasets")

for dataset_name in dataset_names:
    dataset_num = dataset_name[-1]
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)

    data_yaml_path = os.path.join(dataset_path, "data.yaml")

    final_exp_name = EXP_NAME + "_DATASET_" + str(dataset_num)

    final_project_name = EXP_NAME + "_trainings"

    text = f"\
yolo \
mode=train \
task=detect \
project={final_project_name} \
name={final_exp_name} \
model=yolov8n.yaml \
data={data_yaml_path} \
epochs={EPOCH_NUMBER} \
hsv_h=0.0 \
hsv_s=0.0 \
hsv_v=0.0 \
cache=True \
pretrained=False \
batch={BATCH_NUMBER} \
device=0 \
close_mosaic=20 \
patience=0 \
mosaic=0.2 \
"
    

    print(text)
    subprocess.run(text, shell=True)