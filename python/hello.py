#!/usr/bin/env python3
import tensorflow as tf

def init():
    """
    Setup for tensorflow
    """
    # Limit TensorFlow to CPU
    # tf.config.set_visible_devices([], "GPU")
    # tf.device("/CPU:0")

    # Limit TensorFlow to GPU
    # tf.device("/GPU:0")

    print(f"\ntensorflow version is {tf.__version__}")
    print(f"{tf.config.list_physical_devices('GPU')}")

    physical_devices = tf.config.list_physical_devices("GPU")
    logical_devices = tf.config.list_logical_devices()

    print(f"\nNumber of devices: {len(logical_devices)}")

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


# The driver function (confirm that code is under main function)
if __name__ == "__main__":
    init()

    print("\nDone!")