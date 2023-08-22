import subprocess
import glob
import os

DATASETS_ROOT_PATH = "/home/umut/Desktop/newest_thermal_disaster_datasets/hist_eq_filtered_variations"
DEST_PATH = "/home/umut/Desktop/newest_thermal_disaster_datasets/hist_eq_filtered_variations_tars"


for dataset_name in os.listdir(DATASETS_ROOT_PATH):
    print(f"compressing {dataset_name}")
    os.chdir(os.path.join(DATASETS_ROOT_PATH, dataset_name))
    dataset_path = os.path.join(DATASETS_ROOT_PATH, dataset_name, "datasets")
    dest_path = os.path.join(DEST_PATH, dataset_name)
    bash_command = f"tar -czvf {dataset_name}.tar.gz datasets"
    subprocess.run(bash_command.split())

