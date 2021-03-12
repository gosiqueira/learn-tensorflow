"""
Save and load models
---
Model progress can be saved during and after training. This means a model can resume where
it left off and avoid long training times. Saving also means you can share your model and
others can recreate your work. When publishing research models and techniques, most machine
learning practitioners share:
    * code to create the model, and
    * the trained weights, or parameters, for the model
Sharing this data helps others understand how the model works and try it themselves with new data.
---
Original source: https://www.tensorflow.org/tutorials/keras/save_and_load
"""

import os

import tensorflow as tf
from tensorflow.keras import layers


def create_model():
    model = tf.keras.models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )

    return model


def main():
    print(f'TensorFlow version: {tf.__version__}')

    # Get and example dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0001

    # Define a model
    # Define a simple sequential model
    model = create_model()

    # Display the model's architecture
    model.summary()

    # Save checkpoints during training
    # Checkpoint callback usage
    checkpoint_path = 'training_1/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    # Train the model with the new callback
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
        callbacks=[cp_callback] # Pass callback to training
    )

    # This may generate warnings related to saving the state of the optimizer.
    # These warnings (and similar warnings throughout this notebook)
    # are in place to discourage outdated usage, and can be ignored.

    # Create a basic model instance
    model = create_model()

    # Evaluate the model
    _, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Untrained model accuracy: {round(100 * acc, 2)}')

    # Load the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    _, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Restored model accuracy: {round(100 * acc, 2)}')

    # Checkpoint callback options
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    batch_size = 32

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq=5 * batch_size
    )

    # Create a new model instance
    model = create_model()

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the new callback
    model.fit(
        train_images,
        train_labels,
        epochs=50,
        callbacks=[cp_callback],
        validation_data=(test_images, test_labels),
        verbose=0
    )

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    latest

    # Note: the default TensorFlow format only saves the 5 most recent checkpoints.
    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    model.load_weights(latest)

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Restored model accuracy: {round(100 * acc, 2)}')

    # Manually save weights
    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint')

    # Create a new model instance
    model = create_model()

    # Restore the weights
    model.load_weights('./checkpoints/my_checkpoint')

    # Evaluate the model
    _, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Restored model accuracy: {round(100 * acc, 2)}')

    # Save the entire model
    # Create and train a new model instance
    model = create_model()
    model.fit(train_images, train_labels, epochs=5)

    # Save the entire model as a SavedModel.
    os.makedirs('saved_model')
    model.save('saved_model/my_model')

    new_model = tf.keras.models.load_model('saved_model/my_model')

    # Check its architecture
    new_model.summary()

    # Evaluate the restored model
    _, acc = new_model.evaluate(test_images, test_labels, verbose=2)
    print(f'Restored model accuracy {round(100 * acc)}')
    print(f'Preds shape: {new_model.predict(test_images).shape}')

    # HDF5 format
    # Create and train a new model instance
    model = create_model()
    model.fit(train_images, train_labels, epochs=5)

    # Save the entire model to a HDF5 file
    # The '.h5' extension indicates that the model should be saved to HDF5
    model.save('my_model.h5')

    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model('my_model.h5')

    # Show the model architecture
    new_model.summary()

    _, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Restored model accuracy: {round(100 * acc, 2)}')


if __name__ == '__main__':
    main()


# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
