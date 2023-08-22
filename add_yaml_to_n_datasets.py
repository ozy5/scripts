import yaml
import os
import glob

DATASETS_ROOT_PATH = "/home/umut/Desktop/newest_thermal_disaster_datasets/hist_eq_filtered_variations"


for sub_dataset_path in glob.glob(os.path.join(DATASETS_ROOT_PATH, "**", "datasets")):
    for dataset_name in sorted(os.listdir(sub_dataset_path)):
        yaml_dict = {}
        yaml_dict["names"] = ["Human"]
        yaml_dict["nc"] = 1

        yaml_dict["train"] = os.path.join("/datasets", dataset_name, "train")
        yaml_dict["val"] = os.path.join("/datasets", dataset_name, "val")
        yaml_dict["test"] = os.path.join("/datasets", dataset_name, "test")

        with open(os.path.join(sub_dataset_path, dataset_name, "data.yaml"), 'w') as file:
            documents = yaml.dump(yaml_dict, file)
        
        print(f"Created data.yaml for dataset: {dataset_name}")









