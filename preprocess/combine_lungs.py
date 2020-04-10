import cv2
import numpy as np
import os

left_lungs = "/Users/francistoupin/Documents/SYS843/data/masks_left_lung"
right_lungs = "/Users/francistoupin/Documents/SYS843/data/masks_right_lung"
output = "/Users/francistoupin/Documents/SYS843/data/lungs"

if not os.path.exists(output):
    os.mkdir(output)

i = 0
for left_path, right_path in zip(os.listdir(left_lungs), os.listdir(right_lungs)):
    l_img = cv2.imread(os.path.join(left_lungs,left_path), cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(os.path.join(right_lungs,right_path), cv2.IMREAD_GRAYSCALE)

    ret, l_mask = cv2.threshold(l_img, 127, 255, cv2.THRESH_BINARY)
    l_mask = l_mask.astype(np.uint8)
    
    ret, r_mask = cv2.threshold(r_img, 127, 255, cv2.THRESH_BINARY)
    r_mask = r_mask.astype(np.uint8)

    cv2.imwrite(os.path.join(output, left_path), l_mask + r_mask)

    i += 1