import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

from models import build_r2unet
from utils import create_generator
import numpy as np 
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import datetime

def main():
    batch_size = 1
    n_epochs = 10
    TRAINING_SIZE = 192
    TEST_SIZE = 48

    train_generator, _ = create_generator("C:\\Users\\Francis\\Documents\\Repositories\\LungSegmentation\\data\\prepared", add_contours=False, n_augments=batch_size)
    test_generator, _ = create_generator("C:\\Users\\Francis\\Documents\\Repositories\\LungSegmentation\\data\\prepared", add_contours=False, is_train=False, n_augments=1)
    model_checkpoint = ModelCheckpoint('r2unet.hdf5', monitor='loss',verbose=0, save_best_only=True)

    base_log_dir = os.path.join(os.getcwd(), "logs\\fit\\r2unet")
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    log_dir = base_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=log_dir)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

    model = build_r2unet(depth=2,t=2)
    model.fit(train_generator, 
        steps_per_epoch=TRAINING_SIZE,
        epochs=n_epochs,
        verbose=1,
        validation_data=test_generator,
        validation_steps=TEST_SIZE,
        callbacks=[model_checkpoint, early_stopping_callback, tensorboard_callback])


if __name__ == "__main__":
    main()