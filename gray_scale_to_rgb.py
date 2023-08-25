import numpy as np
import cv2

blue_mapping_start_point = 175
red_mapping_start_point = 100

blue_mapping_slope = (255/(255-blue_mapping_start_point))
red_mapping_slope = (255/(255-red_mapping_start_point))

def gray_to_rgb_filter(img):
    return np.stack(((((img-blue_mapping_start_point)*(blue_mapping_slope)) * (img>blue_mapping_start_point)).astype(np.uint8), img, (((img-red_mapping_start_point)*(red_mapping_slope)) * (img>red_mapping_start_point)).astype(np.uint8)), axis=2)

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))

def gray_to_rgb_with_clahe(img): #only to blue and red clannels
    img_clahe = clahe.apply(img)
    return np.stack(((((img_clahe-blue_mapping_start_point)*(blue_mapping_slope)) * (img_clahe>blue_mapping_start_point)).astype(np.uint8), img, (((img_clahe-red_mapping_start_point)*(red_mapping_slope)) * (img_clahe>red_mapping_start_point)).astype(np.uint8)), axis=2)
