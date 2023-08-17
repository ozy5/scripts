import os
from PIL import Image
import numpy as np
import cv2
import time
import math
from gray_scale_to_rgb import gray_to_rgb_filter
import matplotlib.pyplot as plt


DEST_PATH = "GRAY_TO_RGB_FILTERED_IMAGES"

os.makedirs(DEST_PATH, exist_ok=True)

img_path = "imgs/flight2_frame10731.jpg"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# img_max = np.max(img)
# print(f"img_max: {img_max}")

# hist_arr = np.bincount(img.flatten(), minlength=256)
# num_pix = np.sum(hist_arr)
# hist_arr = hist_arr/num_pix
# chist_arr = np.cumsum(hist_arr)

# plt.plot(chist_arr)
# plt.show()


blue_mapping_start_point = 150
red_mapping_start_point = 120

blue_mapping_slope = (255/(255-blue_mapping_start_point))
red_mapping_slope = (255/(255-red_mapping_start_point))

print(f"blue_mapping_slope: {blue_mapping_slope}, red_mapping_slope: {red_mapping_slope}")


start = time.time()



blue_layer = (((img - blue_mapping_start_point)*blue_mapping_slope) * (img>blue_mapping_start_point)).astype(np.uint8)
red_layer = (((img - red_mapping_start_point)*red_mapping_slope) * (img>red_mapping_start_point)).astype(np.uint8)
new_img = np.stack((blue_layer, img, red_layer), axis=2)


print("time taken: ", time.time() - start)

print(f"new_img.shape: {new_img.shape}, img.shape: {img.shape}")



cv2.imshow("img", np.stack((img, img, img), axis=2))
cv2.imshow("new_img", new_img)
cv2.imshow("red_layer_of_new_img", red_layer)
cv2.imshow("green_layer_of_new_img", img)
cv2.imshow("blue_layer_of_new_img", blue_layer)

cv2.waitKey(0)



