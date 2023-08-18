import os
from PIL import Image
import numpy as np
import cv2
import time
import math
from gray_scale_to_rgb import gray_to_rgb_filter
import matplotlib.pyplot as plt
from PIL import Image


DEST_PATH = "/home/umut/Desktop/IEEE_disaster_paper/filtered_imgs_for_scripts"

os.makedirs(DEST_PATH, exist_ok=True)

imgs_path = "/home/umut/Desktop/IEEE_disaster_paper/imgs_for_scripts"


blue_mapping_start_point = 130
red_mapping_start_point = 100

blue_mapping_slope = (255/(255-blue_mapping_start_point))
red_mapping_slope = (255/(255-red_mapping_start_point))

print(f"blue_mapping_slope: {blue_mapping_slope}, red_mapping_slope: {red_mapping_slope}")

passed_times = []

for img_name in os.listdir(imgs_path):
    img_path = os.path.join(imgs_path, img_name)
    # img_max = np.max(img)
    # print(f"img_max: {img_max}")

    # hist_arr = np.bincount(img.flatten(), minlength=256)
    # num_pix = np.sum(hist_arr)
    # hist_arr = hist_arr/num_pix
    # chist_arr = np.cumsum(hist_arr)

    # plt.plot(chist_arr)
    # plt.show()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)




    start = time.time()








    # BASIC GLOBAL THRESHOLDING
    blue_layer = (((img - blue_mapping_start_point)*blue_mapping_slope) * (img>blue_mapping_start_point)).astype(np.uint8)
    red_layer = (((img - red_mapping_start_point)*red_mapping_slope) * (img>red_mapping_start_point)).astype(np.uint8)
    new_img = np.stack((blue_layer, img, red_layer), axis=2)









    # #ADAPTIVE THRESHOLDING
    # blue_layer = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)
    # red_layer = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)
    # new_img = np.stack((blue_layer, img, red_layer), axis=2)





    # #COMBINATION OF ADAPTIVE THRESHOLDING AND GLOBAL THRESHOLDING
    # blue_layer = (((img - blue_mapping_start_point)*blue_mapping_slope) * (img>blue_mapping_start_point)).astype(np.uint8)
    # red_layer = cv2.bitwise_not(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 13))
    # new_img = np.stack((blue_layer, img, red_layer), axis=2)    


    # # OTSU THRESHOLDING
    # blue_layer = (((img - blue_mapping_start_point)*blue_mapping_slope) * (img>blue_mapping_start_point)).astype(np.uint8)
    # red_layer = cv2.bitwise_not(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    # new_img = np.stack((blue_layer, img, red_layer), axis=2)











    passed_time = time.time() - start
    passed_times.append(passed_time)
    print("time taken: ", passed_time)








    # cv2.imshow("img", np.stack((img, img, img), axis=2))
    # cv2.imshow("new_img", new_img)
    # cv2.imshow("red_layer_of_new_img", red_layer)
    # cv2.imshow("green_layer_of_new_img", img)
    # cv2.imshow("blue_layer_of_new_img", blue_layer)

    # save the layers seperately adding suffixes to the original image name indicating the channel
    cv2.imwrite(os.path.join(DEST_PATH, img_name + "_1red.png"), red_layer)
    cv2.imwrite(os.path.join(DEST_PATH, img_name + "_0green.png"), img)
    cv2.imwrite(os.path.join(DEST_PATH, img_name + "_2blue.png"), blue_layer)

    # save the new image
    cv2.imwrite(os.path.join(DEST_PATH, img_name + "_3new.png"), new_img)

#print the average inference time
print(f"average time taken: {np.mean(passed_times)} at inferences")



