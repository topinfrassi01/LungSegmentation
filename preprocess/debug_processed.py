import numpy as np 
import argparse
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2

def load_data(img_train_path, img_test_path, label_train_path, label_test_path):
    x_train = np.load(img_train_path)
    x_test = np.load(img_test_path)

    y_train = np.load(label_train_path)
    y_test = np.load(label_test_path)

    num_classes = np.max(y_train) + 1

    # The loss function requires categorical data.
    y_train = to_categorical(y_train, num_classes)
    
    y_test = to_categorical(y_test, num_classes)
    
    return (x_train, y_train, x_test, y_test)


def train(x_train, y_train, batch_size):
    seed = 73

    generator_args = dict(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        data_format="channels_last"
    )

    datagen_x = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True, **generator_args)

    datagen_y = ImageDataGenerator(**generator_args)

    datagen_x.fit(x_train, augment=True, seed=seed)
    datagen_y.fit(y_train, augment=True, seed=seed)

    generator = zip(datagen_x.flow(x_train, seed=seed, batch_size=batch_size), datagen_y.flow(y_train, seed=seed, batch_size=batch_size))

    i = 0
    for x,y in generator:
        print("Batch {0}".format(i))
        for x_, y_ in zip(x,y):
            cv2.imshow("x", x_)
            cv2.imshow("y", y_ * 255)
            k = cv2.waitKey()
            cv2.destroyAllWindows()

            if k == 13:
                exit()
        i += 1


def main():
    parser = argparse.ArgumentParser(description='See generated images and masks from ImageDataGenerator')
    parser.add_argument('--xtrain', type=str, help='Path where training image .npy dataset is.')
    parser.add_argument('--xtest', type=str, help='Path where testing image .npy dataset is.')
    parser.add_argument('--ytrain', type=str, help='Path where training masks .npy dataset is.')
    parser.add_argument('--ytest', type=str, help='Path where testing masks .npy dataset is.')
    parser.add_argument('--batchsize', type=int, default=32, help="Generator's batch size.")

    args = parser.parse_args()

    #Let's assume data is loaded in a "label encoded" way.
    x_train, y_train, x_test, y_test = load_data(args.xtrain, args.xtest, args.ytrain, args.ytest)

    train(x_train, y_train, args.batchsize)


if __name__ == "__main__":
    main()