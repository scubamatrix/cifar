#!/usr/bin/env python
"""
  cifar10_keras.py

  Train a simple deep CNN on the CIFAR10 small images dataset.

  References
    https://www.tensorflow.org/tutorials/images/cnn
    https://www.tensorflow.org/tutorials/quickstart/advanced
    https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    https://towardsdatascience.com/implementing-a-deep-neural-network-for-the-cifar-10-dataset-c6eb493008a5
    https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from tensorflow import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    # Dropout,
)
from tensorflow.keras.models import Sequential

# from keras.utils import to_categorical


# Global variables
batch_size = None


def init():
    """
    Setup for tensorflow and tensorflow-metal
    """
    print(f"\ntensorflow version is {tf.__version__}")

    # Limit TensorFlow to CPU
    tf.config.set_visible_devices([], "GPU")
    tf.device("/CPU:0")

    # Limit TensorFlow to GPU
    # tf.device("/GPU:0")

    physical_devices = tf.config.list_physical_devices("GPU")
    logical_devices = tf.config.list_logical_devices()

    print(f"\nNum devices: {len(logical_devices)}")
    print(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")

    print("\nList of physical devices:")
    for x in physical_devices:
        print(f"  {x}")

    print("\nList of logical devices:")
    for x in logical_devices:
        print(f"  {x}")

    if tf.config.experimental.list_logical_devices("GPU"):
        print("\nGPU found")
    else:
        print("\nNo GPU found")

    # Find out which devices your operations and tensors are assigned to (True/False)
    tf.debugging.set_log_device_placement(True)

    # The default image data format convention is different on macOS and Linux.
    # NCHW - channel first
    # NHWC - channel last (default)
    print(f"\nimage_data_format: {tf.keras.backend.image_data_format()}")

    # Set image data format to "channels_first"
    # tf.keras.backend.set_image_data_format("channels_first")
    # print(f"image_data_format: {tf.keras.backend.image_data_format()}")

    # # Place tensors on CPU
    # with tf.device('/CPU:0'):
    #   a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #   b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    #
    # # Run on the GPU
    # c = tf.matmul(a, b)
    # print(c)

    print()

    return 0


def show_images():
    """
    Load the CIFAR-10 dataset create a plot of the first nine images
    in the train dataset.
    """
    # load the dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # summarize loaded dataset
    print("Train: X=%s, y=%s" % (X_train.shape, y_train.shape))
    print("Test: X=%s, y=%s" % (X_test.shape, y_test.shape))

    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(X_train[i])

    # show the figure
    plt.show()


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def load_data():
    """
    Load train and test datasets
    """
    # load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # perform train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # one hot encode target values
    # y_train = to_categorical(y_train)
    # y_val = to_categorical(y_val)
    # y_test = to_categorical(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize(train, val, test):
    """
    Normalize the pixels
    """
    # convert from integers to floats
    train_norm = train.astype("float32")
    val_norm = val.astype("float32")
    test_norm = test.astype("float32")

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    val_norm = val_norm / 255.0
    test_norm = test_norm / 255.0

    # return normalized images
    return train_norm, val_norm, test_norm


def create_dataset(X_train, y_train, X_val, y_val, X_test, y_test):
    # Add channels dimension
    X_train = X_train[..., tf.newaxis].astype("float32")
    X_val = X_val[..., tf.newaxis].astype("float32")
    X_test = X_test[..., tf.newaxis].astype("float32")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=10000)
        .batch(batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    print(f"train: {X_train.shape}, {y_train.shape}")
    print(f"val: {X_val.shape}, {y_val.shape}")
    print(f"test: {X_test.shape}, {y_test.shape}\n")

    return train_ds, val_ds, test_ds


def summarize_diagnostics(history):
    """
    Plot diagnostic learning curves
    """
    # plot loss
    plt.subplot(211)
    plt.title("Cross Entropy Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="orange", label="test")

    # plot accuracy
    plt.subplot(212)
    plt.title("Classification Accuracy")
    plt.plot(history.history["accuracy"], color="blue", label="train")
    plt.plot(history.history["val_accuracy"], color="orange", label="test")

    # save plot to file
    filename = sys.argv[0].split("/")[-1] + "_plot.png"
    file_path = os.path.join("../output", filename)

    plt.savefig(file_path)
    plt.close()


def main():
    """
    Run the test harness for evaluating a model
    """
    # load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # prepare pixel data
    X_train, X_val, X_test = normalize(X_train, X_val, X_test)

    # create tf datasets
    train_ds, val_ds, test_ds = create_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    # define model
    model = create_model()

    # We use the EarlyStopping callback to interrupt training
    # when the validation loss is no longer improving
    # es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10)

    # fit model
    history = model.fit(
        train_ds,
        epochs=10,
        batch_size=batch_size,
        # callbacks=[es_callback],
        validation_data=val_ds,
        verbose=1,
    )

    # evaluate model
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print("\n> %.4f " % test_acc)

    # learning curves
    summarize_diagnostics(history)


if __name__ == "__main__":
    batch_size = 32

    # # Create directory if not exist
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)

    init()
    # show_images()
    main()

    print("\nDone!")
