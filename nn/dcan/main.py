from __future__ import division
import argparse
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from model import DCAN


def load_data(img_train_path, img_test_path, label_train_path, label_test_path, contours_train_path, contours_test_path):
    x_train = np.load(img_train_path)
    x_test = np.load(img_test_path)

    masks_train = np.load(label_train_path)
    masks_test = np.load(label_test_path)

    contours_train = np.load(contours_train_path)
    contours_test = np.load(contours_test_path)

    num_classes = np.max(masks_train) + 1

    # The loss function requires categorical data.
    masks_train = to_categorical(masks_train, num_classes)
    masks_test = to_categorical(masks_test, num_classes)
    
    contours_train = to_categorical(contours_train, num_classes)
    contours_test = to_categorical(contours_test, num_classes)

    for d in [x_train, x_test, masks_train, masks_test, contours_train, contours_test]:
        print("Loaded dataset with shape : {0} with dtype {1}".format(d.shape, d.dtype))

    return (x_train, masks_train, contours_train, x_test, masks_test, contours_test)


def train(train_generator, nepochs, batch_size, output):
    
    model = DCAN()
    model.train(train_generator, batch_size, nepochs, output)

    return model

def create_batch_generator(x, masks, contours, batch_size, is_train_data=True):
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

    datagen_masks = ImageDataGenerator(**generator_args)
    datagen_contours = ImageDataGenerator(**generator_args)

    datagen_x.fit(x, augment=is_train_data, seed=seed)
    datagen_masks.fit(masks, augment=is_train_data, seed=seed)
    datagen_contours.fit(contours, augment=is_train_data, seed=seed)

    generator = zip(datagen_x.flow(x, seed=seed, batch_size=batch_size), 
                    datagen_masks.flow(masks, seed=seed, batch_size=batch_size),
                    datagen_contours.flow(contours, seed=seed, batch_size=batch_size))

    return generator


def main():
    parser = argparse.ArgumentParser(description='Run DCAN on specified data')
    parser.add_argument('--xtrain', type=str, help='Path where training image .npy dataset is.')
    parser.add_argument('--xtest', type=str, help='Path where testing image .npy dataset is.')
    parser.add_argument('--maskstrain', type=str, help='Path where training masks .npy dataset is.')
    parser.add_argument('--maskstest', type=str, help='Path where testing masks .npy dataset is.')
    parser.add_argument('--contourstrain', type=str, help='Path where training masks .npy dataset is.')
    parser.add_argument('--contourstest', type=str, help='Path where testing masks .npy dataset is.')
    parser.add_argument('--output', type=str, help="Path of the neural network's output.")
    parser.add_argument('--batchsize', type=int, default=32, help="Batch size. (Defaut 32)")
    parser.add_argument('--nepochs', type=int, default=5, help="Number of epochs. (Default 5)")

    args = parser.parse_args()

    #Let's assume data is loaded in a "label encoded" way.
    x_train, masks_train, contours_train, x_test, masks_test, contours_test = load_data(args.xtrain, args.xtest, args.maskstrain, args.maskstest, args.contourstrain, args.contourstest)

    train_generator = create_batch_generator(x_train, masks_train, contours_train, args.batchsize)
    test_generator = create_batch_generator(x_test, masks_test, contours_test, args.batchsize)

    model = train(train_generator, args.nepochs, args.batch_size, args.output)

    # TODO : Add something to test

if __name__ == "__main__":
    main()
