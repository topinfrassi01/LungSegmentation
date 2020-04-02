from PIL import Image
import numpy as np
import glob
import pandas as pd

from skimage import measure,morphology
from skimage.morphology import disk

__author__ = "Mathias Baltzersen and Rasmus Hvingelby"

class GlandHandler:
    def __init__(self, path):
        self.path = path

        df = pd.read_csv(path + '/Grade.csv')
        df[' grade (GlaS)'] = pd.get_dummies(df[' grade (GlaS)'], drop_first=True)
        df.rename(columns={' grade (GlaS)': 'malignant'}, inplace=True)

        self.data_df = df

    def chop_image(self, image, size=256):
        """
        Chops an image into four quadrants of given size.
        On for each corner of original image.
        :param image:
        :param size:
        :return:
        """
        w, h = image.size

        boxes = [
            (0, 0, size, size),  # Top left corner
            (w - size, 0, w, size),  # Top right corner
            (0, h - size, size, h),  # Bottom left corner
            (w - size, h - size, w, h)  # Bottom right corner
        ]

        return [np.asarray(image.crop(box)) for box in boxes]


    def get_gland(self):

        x_train = []
        y_train_seg = []
        y_train_cont = []
        x_a_test = []
        x_b_test = []
        y_a_test = []
        y_b_test = []

        for filename in sorted(glob.glob(self.path + '/train*.bmp'),
                               key=lambda f: int(filter(lambda x: x.isdigit(), f))):
            img = Image.open(filename)
            # test resizing
            w, h = img.size
            img = img.resize((int(w * 0.5), int(h * 0.5)), Image.ANTIALIAS)  # Shrink the image on test images

            np_img = np.array(img)  # self.chop_image(img, size=shape)

            if 'anno' in filename:
                np_img_cont = self.get_contours(np_img)
                np_img_cont = self.preprocess_annotation(np_img_cont)

                np_img_seg = self.preprocess_annotation(np_img, filename)
                y_train_seg.append(np_img_seg)
                y_train_cont.append(np_img_cont)
            else:
                x_train.append(np_img)

        for filename in sorted(glob.glob(self.path + '/test*.bmp'),
                               key=lambda f: int(filter(lambda x: x.isdigit(), f))):

            img = Image.open(filename)

            img = img.resize((512, 384), Image.ANTIALIAS)  # Shrink the image on test images

            np_img = np.asarray(img)

            if 'testA' in filename and 'anno' not in filename:
                x_a_test.append(np_img)
            elif 'testA' in filename and 'anno' in filename:
                y_a_test.append(np_img)
            elif 'testB' in filename and 'anno' not in filename:
                x_b_test.append(np_img)
            else:
                y_b_test.append(np_img)

        return (np.array(x_train),
                np.array(y_train_seg),
                np.array(y_train_cont),
                np.array(x_a_test),
                np.array(y_a_test),
                np.array(x_b_test),
                np.array(y_b_test))

    def get_contours(self, img):
        im_edges = np.zeros_like(img)

        contours = measure.find_contours(img, 0)

        for cont in contours:
            transpose_cont = np.array(cont.T, dtype=int)
            im_edges[transpose_cont[0], transpose_cont[1]] = 1

        im_edges = morphology.dilation(im_edges, disk(3))

        return im_edges

    def preprocess_annotation(self, img_input, img_name=None):

        img = np.array(img_input)  # Make a new instance of the image

        if img_name is not None:
            img_name = img_name.split('.')[0].split('/')[-1].replace("_anno", "")
            malignant = self.data_df.loc[self.data_df.name == img_name].malignant.values[0]
        else:
            malignant = True

        if not malignant:
            img[img > 0] = 1
        else:
            # this can be changed to work with three classes instead of binary(set to 2 and add another mask)
            img[img > 0] = 1

        img = np.expand_dims(img, axis=3)

        mask0 = np.isin(img, 0).astype(int)
        mask1 = np.isin(img, 1).astype(int)

        img = np.concatenate((mask0, mask1), axis=2)

        return img
