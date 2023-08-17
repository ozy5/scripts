import numpy as np


blue_mapping_start_point = 150
red_mapping_start_point = 120

blue_mapping_slope = (255/(255-blue_mapping_start_point))
red_mapping_slope = (255/(255-red_mapping_start_point))

def gray_to_rgb_filter(img):
    return np.stack(((((img-blue_mapping_start_point)*(blue_mapping_slope)) * (img>blue_mapping_start_point)).astype(np.uint8), img, (((img-red_mapping_start_point)*(red_mapping_slope)) * (img>red_mapping_start_point)).astype(np.uint8)), axis=2)



