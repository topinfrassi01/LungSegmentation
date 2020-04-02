from __future__ import division
import numpy as np
from skimage.transform import rotate
import random

__author__ = "Mathias Baltzersen and Rasmus Hvingelby"


###############################
# Data augmentation for batch #
###############################

def augment_batch(x_train, y_train_seg, y_train_cont, rotation_angle_range):
    x_train_aug = []
    y_train_seg_aug = []
    y_train_cont_aug = []

    for x_img, y_seg_img, y_cont_img in zip(x_train, y_train_seg, y_train_cont):

        rotation_angle = random.uniform(-rotation_angle_range,rotation_angle_range)

        # TODO : Rotate ne marchera pt pas pcq la shape est (256,256,1)
        tmp_x = np.array(rotate(x_img, rotation_angle, preserve_range=True, mode="symetric"), dtype=np.uint8)
        tmp_y_seg = np.array(rotate(y_seg_img, rotation_angle, preserve_range=True, mode="symetric"), dtype=np.uint8)
        tmp_y_cont = np.array(rotate(y_cont_img, rotation_angle, preserve_range=True, mode="symetric"), dtype=np.uint8)

        if np.random.rand() > 0.5:
            tmp_x = tmp_x[:, ::-1, :, :]
            tmp_y_seg = tmp_y_seg[:, ::-1, :, :]
            tmp_y_cont = tmp_y_cont[:, ::-1, :, :]

        x_train_aug.append(tmp_x)
        y_train_seg_aug.append(tmp_y_seg)
        y_train_cont_aug.append(tmp_y_cont)

    return np.array(x_train_aug), np.array(y_train_seg_aug), np.array(y_train_cont_aug)
