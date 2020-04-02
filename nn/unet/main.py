from model import unet 
import numpy as np 
import argparse
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
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


def train(x_train, y_train, batch_size, nepochs, output, debug):
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

    generator = zip(datagen_x.flow(x_train, seed=seed), datagen_y.flow(y_train, seed=seed))

    model = unet()
    #model_checkpoint = ModelCheckpoint(os.path.join(output, 'unet.hdf5'), monitor='loss',verbose=1, save_best_only=True)

    if debug:
        for x,y in generator:
            cv2.imshow("x", x[0])
            cv2.imshow("y", y[0] * 255)
            k = cv2.waitKey()
            cv2.destroyAllWindows()

            if k == 13:
                exit()

    model.fit_generator(generator,
                    steps_per_epoch=len(x_train) / batch_size,
                    epochs=nepochs)
                    #callbacks=[model_checkpoint])

    datagen_x.standardize(x_train)
    result = model.predict(x_train, batch_size=1)
    np.save("result.npy", result)
    return model

def main():
    parser = argparse.ArgumentParser(description='Run U-net on specified data')
    parser.add_argument('--xtrain', type=str, help='Path where training image .npy dataset is.')
    parser.add_argument('--xtest', type=str, help='Path where testing image .npy dataset is.')
    parser.add_argument('--ytrain', type=str, help='Path where training masks .npy dataset is.')
    parser.add_argument('--ytest', type=str, help='Path where testing masks .npy dataset is.')
    parser.add_argument('--output', type=str, help="Path of the neural network's output.")
    parser.add_argument('--batchsize', type=int, default=32, help="UNet's batch size.")
    parser.add_argument('--nepochs', type=int, help="UNet's number of epochs.")
    parser.add_argument('--debug', action="store_true", help="Show rotation of training images/masks.")

    args = parser.parse_args()

    #Let's assume data is loaded in a "label encoded" way.
    x_train, y_train, x_test, y_test = load_data(args.xtrain, args.xtest, args.ytrain, args.ytest)

    train(x_train, y_train, args.batchsize, args.nepochs, args.output, args.debug)


if __name__ == "__main__":
    main()