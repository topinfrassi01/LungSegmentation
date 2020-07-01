import numpy as np 
import argparse
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

RANDOM_SEED = 73

def create_generator(path, add_contours, is_train=True, n_augments=1):
    """
    tood
    """
    generator_args = dict(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        data_format="channels_last"
    ) if is_train else {}

    npy_suffix = "train" if is_train else "test"
    img_train = np.load(os.path.join(path, "images_train.npy"))

    images = np.load(os.path.join(path, "images_{0}.npy".format(npy_suffix)))
    generator_img = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, **generator_args)
    generator_img.fit(img_train, augment=is_train, seed=RANDOM_SEED)

    img_flow = generator_img.flow(images, batch_size=n_augments, seed=RANDOM_SEED)

    masks = np.load(os.path.join(path, "masks_{0}.npy".format(npy_suffix)))
    generator_mask = ImageDataGenerator(**generator_args)
    generator_mask.fit(masks, augment=is_train, seed=RANDOM_SEED)
    mask_flow = generator_mask.flow(masks, batch_size=n_augments, seed=RANDOM_SEED)

    if add_contours:
        contours = np.load(os.path.join(path, "contours_{0}.npy".format(npy_suffix)))
        generator_contours = ImageDataGenerator(**generator_args)
        generator_contours.fit(contours, augment=is_train, seed=RANDOM_SEED)
        contours_flow = generator_contours.flow(contours, batch_size=n_augments, seed=RANDOM_SEED)
        
        return zip(img_flow, zip(mask_flow, contours_flow)), lambda x : generator_img.standardize(x)

    return zip(img_flow, mask_flow), lambda x : generator_img.standardize(x)