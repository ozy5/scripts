import os
import shutil
import random


RANDOM_DATASET_COUNT = 10

DATASET_SOURCE_PATH = "/home/umut/Desktop/newest_thermal_disaster_datasets/dataset_source_not_filtered"

DEST_PATH = "/home/umut/Desktop/newest_thermal_disaster_datasets/not_filtered_random_datasets"

dataset_path_images = os.path.join(DATASET_SOURCE_PATH, "images")
dataset_path_labels = os.path.join(DATASET_SOURCE_PATH, "labels")



train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

all_names_images = sorted(os.listdir(dataset_path_images))
all_names_labels = sorted(os.listdir(dataset_path_labels))


for name, labels in zip(all_names_images, all_names_labels):
    if(name.split(".jpg")[0] != labels.split(".txt")[0]):
        print("ERROR")
        break


dataset_size = len(all_names_images)

print(f"Dataset has {dataset_size} labeles images.\nNow, creating {RANDOM_DATASET_COUNT} datasets at the location: {DEST_PATH}")






for dataset_number in range(RANDOM_DATASET_COUNT):

    random_seed = dataset_number
    random.seed(random_seed)

    DATA_SET_DEST_PATH = os.path.join(DEST_PATH, "not_filtered_dataset_" + str(dataset_number))

    os.makedirs(DATA_SET_DEST_PATH, exist_ok=True)

    DATA_SET_DEST_PATH_TRAIN = os.path.join(DATA_SET_DEST_PATH, "train")
    DATA_SET_DEST_PATH_VAL = os.path.join(DATA_SET_DEST_PATH, "val")
    DATA_SET_DEST_PATH_TEST = os.path.join(DATA_SET_DEST_PATH, "test")

    DATA_SET_DEST_PATH_TRAIN_IMAGES = os.path.join(DATA_SET_DEST_PATH_TRAIN, "images")
    DATA_SET_DEST_PATH_TRAIN_LABELS = os.path.join(DATA_SET_DEST_PATH_TRAIN, "labels")

    DATA_SET_DEST_PATH_VAL_IMAGES = os.path.join(DATA_SET_DEST_PATH_VAL, "images")
    DATA_SET_DEST_PATH_VAL_LABELS = os.path.join(DATA_SET_DEST_PATH_VAL, "labels")

    DATA_SET_DEST_PATH_TEST_IMAGES = os.path.join(DATA_SET_DEST_PATH_TEST, "images")
    DATA_SET_DEST_PATH_TEST_LABELS = os.path.join(DATA_SET_DEST_PATH_TEST, "labels")

    os.makedirs(DATA_SET_DEST_PATH_TRAIN_IMAGES, exist_ok=True)
    os.makedirs(DATA_SET_DEST_PATH_TRAIN_LABELS, exist_ok=True)

    os.makedirs(DATA_SET_DEST_PATH_VAL_IMAGES, exist_ok=True)
    os.makedirs(DATA_SET_DEST_PATH_VAL_LABELS, exist_ok=True)

    os.makedirs(DATA_SET_DEST_PATH_TEST_IMAGES, exist_ok=True)
    os.makedirs(DATA_SET_DEST_PATH_TEST_LABELS, exist_ok=True)




    random_index_list = list(range(dataset_size))



    random.shuffle(random_index_list)

    train_index_list = random_index_list[0 : int(train_ratio * dataset_size)]
    val_index_list = random_index_list[int(train_ratio * dataset_size) : int((train_ratio + val_ratio) * dataset_size)]
    test_index_list = random_index_list[int((train_ratio + val_ratio) * dataset_size) : ]


    for index in train_index_list:
        shutil.copy(os.path.join(dataset_path_images, all_names_images[index]), DATA_SET_DEST_PATH_TRAIN_IMAGES)
        shutil.copy(os.path.join(dataset_path_labels, all_names_labels[index]), DATA_SET_DEST_PATH_TRAIN_LABELS)

    for index in val_index_list:
        shutil.copy(os.path.join(dataset_path_images, all_names_images[index]), DATA_SET_DEST_PATH_VAL_IMAGES)
        shutil.copy(os.path.join(dataset_path_labels, all_names_labels[index]), DATA_SET_DEST_PATH_VAL_LABELS)

    for index in test_index_list:
        shutil.copy(os.path.join(dataset_path_images, all_names_images[index]), DATA_SET_DEST_PATH_TEST_IMAGES)
        shutil.copy(os.path.join(dataset_path_labels, all_names_labels[index]), DATA_SET_DEST_PATH_TEST_LABELS)


