#!/usr/bin/env python3
"""
  cifar10.py

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

import argparse
import math
import os
import sys
import time

from datetime import datetime
from math import floor
from pprint import pprint
from sklearn.model_selection import train_test_split

from tabulate import tabulate

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD

# from tensorflow.preprocessing.image import ImageDataGenerator


class Logger(object):
    """
    This helper class allows the application to write
    to both stdout and a file at the same time.
    """

    # Initializing (constructor)
    def __init__(self, name):
        self.count = 0
        self.terminal = sys.stdout
        self.log = open(name, "a")

    # Deleting (destructor)
    def __del__(self):
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.count += 1

        # Handle long-running processes
        if self.count == 100:
            self.flush()
            self.count = 0

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This method handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        # pass
        self.log.flush()


class Params:
    """
    Program parameters
    """

    # stream = 'cse'     # Class variable

    def __init__(self):
        # prog_name = os.path.basename(__file__)
        # basename, extension = os.path.splitext(prog_name)
        # self.prog_name = basename

        # Instance variables
        self.debug = False
        self.use_logger = False

        self.model_type = None
        self.prog_name = None

        self.data_dir = "../data"
        self.model_dir = "../models"
        self.output_dir = "../output"

        prog_name = os.path.basename(__file__)
        basename, extension = os.path.splitext(prog_name)
        self.prog_name = basename
        self.log_file = "{0}/{1}.txt".format(self.output_dir, basename)

        # training parameters
        self.num_epochs = 200
        self.num_trials = 10
        self.patience = 10

        # model parameters
        self.loss = "categorical_crossentropy"
        self.objective = "val_loss"
        self.optimizer = "adam"
        self.metrics = ["acc"]

        # model hyperparameters
        self.batch_size = 32      # 128
        self.inputs = None
        self.units = None          # 32
        self.activation = None  # relu
        self.dropout = None
        self.lr = None            # 0.001
        self.weight_decay = None  # 1e-4

        # lr_init = 0.01
        # self.weight_decay = self.lr / self.num_epochs


    def __repr__(self):
        return "<{0} num_steps_in: {1} num_features: {2} num_steps_out: {3}>".format(
            self.__class__.__name__,
            self.num_steps_in,
            self.num_features,
            self.num_steps_out,
        )

    def __str__(self):
        ignore_list = ["dict_tuner", "inputs", "targets"]
        d_vars = vars(self)
        s_out = ""

        keys_list = list(d_vars.keys())
        values_list = list(d_vars.values())
        print(f"d_vars: {len(d_vars.items())}")

        # find longest key
        key_len = [len(x) for x in d_vars.keys()]
        df = pd.DataFrame()

        for i, (key, value) in enumerate(d_vars.items()):
            if key not in ignore_list:
                df_new_item = pd.DataFrame(
                    {
                        "name": [
                            str(keys_list[i]),
                        ],
                        "value": [
                            str(values_list[i]),
                        ],
                    }
                )
                df = pd.concat([df, df_new_item], axis=0, ignore_index=True)

        # display the DataFrame
        s_out = tabulate(df, headers="keys", tablefmt="plain")

        return s_out

    def get_file_path(self, file_name):
        prog_name = os.path.basename(__file__)
        basename, extension = os.path.splitext(prog_name)
        params.prog_name = basename
        file_out = "{0}/{1}.txt".format(self.output_dir, basename)

        return file_out


class Timer(object):
    """
    This is a helper class for time calculations.
    """

    def __init__(self):
        self.start = time.time()
        self.end = None
        self.diff = None

    def elapsed_time(self):
        """
        Compute elapsed time.

        @param start: start time
        @param end:   end time
        @return:      elapsed time (string)
        """
        days, hours, minutes = [0, 0, 0]
        s_time = ""

        self.end = time.time()
        diff = self.end - self.start
        self.diff = diff

        if diff > 86400:  # day
            days = floor(diff / 86400)
            diff = diff - days * 86400
        if diff > 3600:  # hour
            hours = floor(diff / 3600)
            diff = diff - hours * 3600
        if diff > 60:  # minute
            minutes = floor(diff / 60)
            diff = diff - minutes * 60

        if days > 0:
            s_time = "{0} days {1} hrs {2} min {3:.4f} sec".format(days, hours, minutes, diff)
            # print(f"{days} days {hours} hrs {minutes} min {diff:.4f} sec")
        elif hours > 0:
            s_time = "{0} hrs {1} min {2:.4f} sec".format(hours, minutes, diff)
            # print(f"{hours} hrs {minutes} min {diff:.4f} sec")
        elif minutes > 0:
            s_time = "{0} min {1:.4f} sec".format(minutes, diff)
            # print(f"{minutes} min {diff:.4f} sec")
        else:
            s_time = "{0:.4f} sec".format(diff)
            # print(f"{diff: .4f} sec")

        return s_time


def init():
    init_tf()
    # init_ray()
    return 0


def init_tf():
    """
    Setup for tensorflow and tensorflow-metal
    """
    # Limit TensorFlow to CPU
    # tf.config.set_visible_devices([], "GPU")
    # tf.device("/CPU:0")

    # Limit TensorFlow to GPU
    tf.device("/GPU:0")

    # tf.config.set_visible_devices([], "GPU")
    # with tf.device("/CPU:0"):

    # Limit TensorFlow to GPU
    # with tf.device("/GPU:0"):

    physical_devices = tf.config.list_physical_devices("GPU")
    logical_devices = tf.config.list_logical_devices()

    print(f"\ntensorflow version is {tf.__version__}")

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

    # Find out which devices your operations and tensors are assigned to
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


def init_ray():
    """
    Setup for ray
    There are errors using ray
    """
    os.environ["MODIN_CPUS"] = "4"
    os.environ["MODIN_ENGINE"] = "ray"
    os.environ["MODIN_MEMORY"] = "5000000000"  # 5*10^9 = 5GB
    os.environ["MODIN_NPARTITIONS"] = "2"
    os.environ["MODIN_PROGRESS_BAR"] = "true"
    os.environ["MODIN_STORAGE_FORMAT"] = "Pandas"
    os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

    # ProgressBar.enable()

    return 0


def show_images():
    """
    Show samples from each class
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_train, img_channels, img_rows, img_cols = X_train.shape
    num_test, _, _, _ = X_train.shape
    num_classes = len(np.unique(y_train))

    # If using tensorflow, set image dimensions order
    if K.backend() == "tensorflow":
        K.common.set_image_dim_ordering("th")

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    fig = plt.figure(figsize=(8, 3))

    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:] == i)[0]
        x_idx = X_train[idx, ::]
        img_num = np.random.randint(x_idx.shape[0])
        im = np.transpose(x_idx[img_num, ::], (1, 2, 0))
        ax.set_title(class_names[i])
        plt.imshow(im)

    plt.show()


def elapsed_time(start, end):
    """
    Compute elapsed time.

    @param start: start time
    @param end:   end time
    @return:      elapsed time (string)
    """
    diff = end - start
    days, hours, minutes = [0, 0, 0]
    s_time = []

    if diff > 86400:  # day
        days = math.floor(diff / 86400)
        diff = diff - days * 86400
    if diff > 3600:  # hour
        hours = math.floor(diff / 3600)
        diff = diff - hours * 3600
    if diff > 60:  # minute
        minutes = math.floor(diff / 60)
        diff = diff - minutes * 60

    if days > 0:
        s_time = "{0} days {1} hrs {2} min {3:.4f} sec".format(days, hours, minutes, diff)
        # print(f"{days} days {hours} hrs {minutes} min {diff:.4f} sec")
    elif hours > 0:
        s_time = "{0} hrs {1} min {2:.4f} sec".format(hours, minutes, diff)
        # print(f"{hours} hrs {minutes} min {diff:.4f} sec")
    elif minutes > 0:
        s_time = "{0} min {1:.4f} sec".format(minutes, diff)
        # print(f"{minutes} min {diff:.4f} sec")
    else:
        s_time = "{0:.4f} sec".format(diff)
        # print(f"{diff: .4f} sec")

    return s_time


def timestamp():
    """
    Compute timestamp

    @return:
    """
    # Calling now() function
    today = datetime.now()

    s_timestamp = "{0}{1:02d}{02:02d}-{3:02d}{4:02d}{5:02d}".format(
        today.year, today.month, today.day, today.hour, today.minute, today.second
    )

    return s_timestamp


def accuracy(test_x, test_y, model):
    """
    Compute test accuracy

    @param test_x:
    @param test_y:
    @param model:
    @return:
    """
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct) / result.shape[0]

    return accuracy * 100


def plot_history(model_history):
    """
    Plot model accuracy and loss
    @param model_history:
    @return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history["acc"]) + 1), model_history.history["acc"])
    axs[0].plot(
        range(1, len(model_history.history["val_acc"]) + 1),
        model_history.history["val_acc"],
    )
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_xticks(
        np.arange(1, len(model_history.history["acc"]) + 1),
        len(model_history.history["acc"]) / 10,
    )
    axs[0].legend(["train", "val"], loc="best")

    # Summarize history for loss
    axs[1].plot(range(1, len(model_history.history["loss"]) + 1), model_history.history["loss"])
    axs[1].plot(
        range(1, len(model_history.history[params.objective]) + 1),
        model_history.history[params.objective],
    )
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_xticks(
        np.arange(1, len(model_history.history["loss"]) + 1),
        len(model_history.history["loss"]) / 10,
    )
    axs[1].legend(["train", "val"], loc="best")

    filename = "model-history-" + timestamp() + ".png"

    file_path = os.path.join(params.output_dir, filename)

    plt.savefig(file_path)  # Save plot to file
    # plt.show()            # Show plot
    plt.clf()               # Clear current figure
    plt.close(fig)


def load_data(params: Params):
    """
    Load train and test datasets
    """
    # load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # perform train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # params.num_classes = y_train.shape[1]

    # one-hot encode target values (not needed when from_logits=True)
    if params.loss == "categorical_crossentropy":
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        y_test = to_categorical(y_test)

    if params.debug:
        print("\n=== load_data ===")

        print(f"\nnum_classes = {y_train.shape[1]}")

        print("\nX_train shape:", X_train.shape)
        print("X_valid shape:", X_val.shape)
        print("X_test shape:", X_test.shape)

        print(f"\ny_train shape: {y_train.shape}")
        print(f"y_valid shape: {y_val.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Data Sanity Check
        print(f"\nnum_classes = {y_test.shape[1]}")
        print(f"X_test shape: {X_test.shape}")
        # print(f"y_test:\n{y_test[0:10]}")  # Check that dataset has been randomized

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
        .batch(params.batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(params.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(params.batch_size)

    print(f"train: {X_train.shape}, {y_train.shape}")
    print(f"val: {X_val.shape}, {y_val.shape}")
    print(f"test: {X_test.shape}, {y_test.shape}\n")

    return train_ds, val_ds, test_ds


def create_npz_dataset(params: Params):
    """
    Data pre-processing

    @return:
    """
    # Create train/test/validation split
    # We want to split our dataset into separate training and test datasets
    # We use the training dataset to fit the model and the test dataset to evaluate
    # its performance to generalize to unseen data.
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.5)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_valid = X_valid.astype("float32")

    # Standardize the columns
    # We need to standardize the columns before we feed them to a linear classifier,
    # but if the X values are in the range 0-255 then we can transform them to [0,1].
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_valid = X_valid / 255.0

    # One-hot encoding
    # Represent each integer value as a binary vector that is all zeros
    # except the index of the integer.
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    y_valid_hot = to_categorical(y_valid)

    # y_categories = np.unique(y_train)

    # Save datasets to file
    np.savez(
        "cifar10_test.npz",
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        y_train_hot=y_train_hot,
        y_valid_hot=y_valid_hot,
        y_test_hot=y_test_hot,
    )


def create_model(params: Params, name):
    """
    Create model for 70% accuracy
    The parameters needed to be different to run on linux.
    @return:
    """
    params.model_name = name

    model_json = "{0}/{1}.json".format(params.output_dir, name)
    params.model_json = os.path.join(params.output_dir, model_json)

    model_file = "{0}/{1}.keras".format(params.output_dir, name)
    params.model_file = os.path.join(params.output_dir, model_file)

    # We expect our inputs to be RGB images of arbitrary size
    inputs = keras.Input(shape=(32, 32, 3))

    # Create the model
    x = Sequential()(inputs)

    # Apply some convolution and pooling layers
    x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)

    # Add a dense classifier on top
    if params.loss != "categorical_crossentropy":
        outputs = Dense(10)(x)                        # SparseCategoricalCrossentropy
    else:
        # The activation is softmax which makes the output sum up to 1
        # so the output can be interpreted as probabilities.
        # The model will then make predictions based on which option has a higher probability.
        outputs = Dense(10, activation="softmax")(x)  # categorical_crossentropy

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    # We can print a summary of how your data gets transformed at each stage of the model.
    # This is useful for debugging.
    model.summary()

    # We can also plot the model as a graph.
    # keras.utils.plot_model(model, "first_model.png")

    # Compile the model
    model.compile(
        optimizer=params.optimizer,
        loss=params.loss,
        metrics=["accuracy"],
    )

    return model


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


def fit_model(params: Params, model, train_ds, val_ds):
    """
    Train the model
    @param model:
    @return:
    """
    timer = Timer()

    # We use the EarlyStopping callback to interrupt training
    # when the validation loss is no longer improving
    # es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10)

    # Train the model
    # The history object which records what happened over the course of training.
    # The history.history dict contains per-epoch time-series of metrics values.
    history = model.fit(
        train_ds,
        epochs=params.num_epochs,
        batch_size=params.batch_size,
        validation_data=val_ds,
        verbose=1,
    )

    print("\nFinished training!")
    print(f"Model took [{timer.elapsed_time()}] to train")

    # Save the model architecture to disk
    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    model_json = model.to_json()
    with open(params.model_json, "w") as json_file:
        json_file.write(model_json)

    # del model  # deletes the existing model

    # Save the model weights
    # model.save_weights("model_cnn.h5")

    # Save the whole model (architecture + weights + optimizer state)
    model.save(params.model_file)  # creates a HDF5 file

    # Plot model history
    # plot_history(history)


def evaluate_model(params: Params, X_train, X_val, X_test, y_train, y_val, y_test):
    model_file = params.model_file

    print("\n=== evaluate_model ===")

    # Return a compiled model (identical to the previous one)
    model = load_model(model_file)

    scores = model.evaluate(X_train, y_train, batch_size=params.batch_size, verbose=0)
    print(f"\nTrain: accuracy: {scores[1]:.4f}  loss: {scores[0]:.4f}")

    loss, acc = model.evaluate(X_val, y_val, batch_size=params.batch_size, verbose=0)
    print(f"Val: accuracy: {acc:.4f}  loss: {loss:.4f}")

    loss, acc = model.evaluate(X_test, y_test, batch_size=params.batch_size, verbose=0)
    print(f"Test: accuracy: {acc:.4f}  loss: {loss:.4f}")

    # x = dict(zip(model.metrics_names, scores))
    # pprint(x, compact=True)

    # Compute test accuracy
    # print(f"\nAccuracy on test data is: {accuracy(X_test, y_test, model):.4f}")

    # Generate predictions (probabilities -- the output of the last layer) on new data.
    print("\nGenerate predictions for 10 samples")
    pred = model.predict(X_test[:10])
    obs = y_test[:10]

    # pred = model.predict(X_test)
    # obs = y_test

    # pred.shape is expected to be (batch_size, num_classes)
    print(f"pred.shape: {pred.shape}, obs.shape: {obs.shape}")

    # pprint(pred, compact=True)
    # pprint(obs)

    # for i in range(10):
    #     y_pred = np.argmax(pred[i])
    #     y_obs = np.argmax(obs[i])
    #     print(f"(y_pred, y_obs) is {y_pred, y_obs}")


def main(params: Params):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(params)

    # Get script filename instead of path
    file_name = os.path.basename(sys.argv[0])
    msg = file_name

    # Initialize parser
    parser = argparse.ArgumentParser(description=msg)

    # Add optional arguments
    parser.add_argument("-i", "--images", dest="images", action="store_true", help="Show images")
    parser.add_argument(
        "-p",
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Preprocess data",
    )
    parser.add_argument("-c", "--create", dest="create", action="store_true", help="Create model")
    parser.add_argument(
        "-a",
        "--augment",
        dest="augment",
        action="store_true",
        help="Data augmentation model",
    )
    parser.add_argument(
        "-e", "--evaluate", dest="evaluate", action="store_true", help="Evaluate model"
    )

    # Read arguments from command line
    args = parser.parse_args()

    if args.images:
        show_images()
    elif args.create:
        create_model()
    elif args.evaluate:
        evaluate_model()
    else:
        print("\nTrain and evaluate model ...\n")

        # prepare pixel data
        X_train, X_val, X_test = normalize(X_train, X_val, X_test)

        # create tf datasets
        train_ds, val_ds, test_ds = create_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

        model = create_model(params, "cifar10_cnn")
        fit_model(params, model, train_ds, val_ds)
        evaluate_model(params, X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    init()

    params = Params()
    params.debug = True
    params.use_logger = True
    params.num_epochs = 15

    model = 2
    if model == 1:
        params.loss = "categorical_crossentropy"
        params.optimizer = SGD(learning_rate=0.01, momentum=0.9)
        # params.optimizer = "sgd"
    else:
        params.loss = SparseCategoricalCrossentropy(from_logits=True)
        # params.optimizer = "adamax"
        # params.optimizer = "rmsprop"
        params.optimizer = "adam"   # adam does not converge on macos with GPU

    # Create directory if not exist
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)

    # Delete log file if it exists
    if os.path.exists(params.log_file):
        os.remove(params.log_file)

    if params.use_logger:
        logger = Logger(params.log_file)
        sys.stdout = logger
        sys.stderr = logger

    if params.debug:
        print("\nparams:")
        print(params)

    main(params)

    print("\nDone!")
