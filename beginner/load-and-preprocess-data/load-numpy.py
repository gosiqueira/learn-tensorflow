"""
Load NumPy data
---
This tutorial provides an example of loading data from NumPy arrays into a `tf.data.Dataset`.
This example loads the MNIST dataset from a .npz file. However, the source of the NumPy
arrays is not important.
---
Original source: https://www.tensorflow.org/tutorials/load_data/numpy
"""

import numpy as np
import tensorflow as tf


def main():
    # Load from .npz file
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    path = tf.keras.utils.get_file('mnist.npz', url)
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    # Load NumPy arrays with `tf.data.Dataset`
    train_ds = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    # Use the datasets
    # Shuffle and batch the datasets
    batch_size = 64
    shuffle_buffer_size = 100

    train_ds = train_ds.shuffle(shuffle_buffer_size).batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    # Build and train a model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )

    model.fit(train_ds, epochs=10)

    model.evaluate(test_ds)


if __name__ == '__main__':
    main()
