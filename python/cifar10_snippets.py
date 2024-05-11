#!/usr/bin/env python3
"""
  cifar10_snippets.py
  Code snippets for CIFAR-10 dataset
"""
import numpy as np
import time
# import matplotlib.pyplot as plt

from cifar10 import Params, accuracy, elapsed_time, load_model, plot_history

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay,
    InverseTimeDecay,
    PolynomialDecay,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate


def load_data():
    """
    Reduce datasets to improve performance
    """
    # Load data from file
    npzfile = np.load("cifar10.npz")
    # print(npzfile.files)

    X_train = npzfile["X_train"]
    X_valid = npzfile["X_valid"]
    X_test = npzfile["X_test"]

    y_train = npzfile["y_train_hot"]
    y_valid = npzfile["y_valid_hot"]
    y_test = npzfile["y_test_hot"]

    num_train, img_channels, img_rows, img_cols = X_train.shape
    num_test, _, _, _ = X_test.shape
    num_classes = y_train.shape[1]
    # num_classes = len(np.unique(y_train))

    print("load_data:")
    print("X_train.shape:", num_train, img_channels, img_rows, img_cols)
    print()

    # Reduce datasets to improve performance
    num_records = 2000
    X_train = X_train[:num_records]
    X_valid = X_valid[:num_records]
    X_test = X_test[:num_records]

    y_train = y_train[:num_records]
    y_valid = y_valid[:num_records]
    y_test = y_test[:num_records]

    print(f"num_classes = {num_classes}")

    print("\nX_train shape:", X_train.shape)
    print("X_valid shape:", X_valid.shape)
    print("X_test shape:", X_test.shape)

    print(f"\ny_train shape: {y_train.shape}")
    print(f"y_valid shape: {y_valid.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Data Sanity Check
    print(f"\nnum_classes = {y_test.shape[1]}")
    print(f"X_test shape: {X_test.shape}")
    # print(f"y_test:\n{y_test[0:10]}")  # Check that dataset has been randomized

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_npz():
    """
    Load data from npz file
    """
    npzfile = np.load("cifar10.npz")
    # print(npzfile.files)

    X_train = npzfile["X_train"]
    X_valid = npzfile["X_valid"]
    X_test = npzfile["X_test"]

    y_train = npzfile["y_train_hot"]
    y_valid = npzfile["y_valid_hot"]
    y_test = npzfile["y_test_hot"]

    num_train, img_channels, img_rows, img_cols = X_train.shape
    num_test, _, _, _ = X_test.shape
    # num_classes = y_train.shape[1]
    # num_classes = len(np.unique(y_train))

    print("load_data:")
    print("X_train.shape:", num_train, img_channels, img_rows, img_cols)
    print()

    # Reduce datasets to improve performance
    # num_records = 2000
    # X_train = X_train[:num_records]
    # X_valid = X_valid[:num_records]
    # X_test = X_test[:num_records]

    # y_train = y_train[:num_records]
    # y_valid = y_valid[:num_records]
    # y_test = y_test[:num_records]

    # num_classes = y_train.shape[1]
    # print(f"num_classes = {num_classes}")
    # print(f"X_test shape: {X_valid.shape}")
    # print(f"y_test shape: {y_valid.shape}")

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_npz_linux():
    """
    Load data from npz file on Linux
    """
    # Load data from file
    npzfile = np.load("cifar10.npz")
    # print(npzfile.files)

    X_train = npzfile["X_train"]
    X_valid = npzfile["X_valid"]
    X_test = npzfile["X_test"]

    y_train = npzfile["y_train_hot"]
    y_valid = npzfile["y_valid_hot"]
    y_test = npzfile["y_test_hot"]

    # Reduce datasets to improve performance
    # num_records = 2000
    # X_train = X_train[:num_records]
    # X_valid = X_valid[:num_records]
    # X_test = X_test[:num_records]

    # y_train = y_train[:num_records]
    # y_valid = y_valid[:num_records]
    # y_test = y_test[:num_records]

    num_train, img_rows, img_cols, img_channels = X_train.shape
    num_test, _, _, _ = X_test.shape
    # num_classes = y_train.shape[1]
    # num_classes = len(np.unique(y_train))

    print("X_train.shape:", num_train, img_channels, img_rows, img_cols)
    print()

    # Convert from NCHW to NHWC
    @tf.function
    def transform(x):
        y = tf.transpose(x, [0, 3, 1, 2])
        return y

    X_train = transform(X_train)
    X_valid = transform(X_valid)
    X_test = transform(X_test)

    print("After transform:")
    print(
        "X_train.shape:", X_train.get_shape()
    )  # the shape of out is [2000, 32, 32, 3]
    print("X_valid.shape:", X_valid.get_shape())
    print("X_test.shape:", X_test.get_shape())

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def data_augment(params: Params):
    """
    Add data augmentation for 86% accuracy

    @return:
    """
    # Load data from file
    npzfile = np.load("cifar10.npz")
    print(npzfile.files)

    X_train = npzfile["X_train"]
    X_valid = npzfile["X_valid"]
    y_train = npzfile["y_train_hot"]
    y_valid = npzfile["y_valid_hot"]

    # Return a compiled model (identical to the previous one)
    model = load_model("model_cnn_70.h5")

    datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)

    # Train the model
    start = time.time()
    history = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=params.batch_size),
        samples_per_epoch=X_train.shape[0],
        epochs=params.num_epochs,
        validation_data=(X_valid, y_valid),
        verbose=1,
    )
    end = time.time()
    print(f"Model took [{elapsed_time(start, end)}] to train (data augmentation)")

    # Save the model architecture to disk
    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    model_json = model.to_json()
    with open("model_cnn.json", "w") as json_file:
        json_file.write(model_json)

    # Save the model weights
    model.save_weights("model_cnn.h5")

    # Save the whole model (architecture + weights + optimizer state)
    model.save("model_cnn.h5")  # creates a HDF5 file

    # Plot model history
    plot_history(history)

    # Compute validation accuracy
    print(
        f"Accuracy on validation data is: {accuracy(X_valid, y_valid, model): .2f} %0.2f"
    )


def fit_sgd_model():
    """
    Sample code for training models
    """
    # lr_init = 0.01
    # decay = lrate / epochs

    lr_init = 0.1
    lr_schedule = ExponentialDecay(
        lr_init, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    lr_init = 0.1
    decay_steps = 1.0
    decay_rate = 0.5
    lr_schedule = InverseTimeDecay(
        lr_init, decay_steps, decay_rate, staircase=False, name=None
    )

    lr_start = 0.1
    lr_end = 0.01
    decay_steps = 10000
    lr_schedule = PolynomialDecay(lr_start, decay_steps, lr_end, power=0.5)

    # sgd = SGD(learning_rate=lr_schedule)
    sgd = SGD(learning_rate=lr_schedule, momentum=0.9)

    # Return a compiled model (identical to the previous one)
    model = load_model("model_cnn.h5")

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])

    # Train the model
    start = time.time()
    # The history object which records what happened over the course of training.
    # The history.history dict contains per-epoch timeseries of metrics values.
    # history = model.fit(
    #     X_train,
    #     y_train,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     validation_data=(X_valid, y_valid),
    #     verbose=1,
    # )
    end = time.time()
    print(f"\nModel took [{elapsed_time(start, end)}] to train")

    # Save the model architecture to disk
    model_json = model.to_json()
    with open("model_cnn.json", "w") as json_file:
        json_file.write(model_json)

    # del model  # delete the existing model

    # Save the model weights
    model.save_weights("model_cnn.h5")

    # Save the whole model (architecture + weights + optimizer state)
    model.save("model_cnn.h5")  # creates a HDF5 file

    # Plot model history
    # plot_history(history)


if __name__ == "__main__":
    params = Params()
    # main()

    print("\nDone!")
