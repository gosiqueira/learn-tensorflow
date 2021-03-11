"""
Text classification with TensorFlow Hub: Movie reviews
---
This script classifies movie reviews as positive or negative using the text of the review.
This is an example of binary—or two-class—classification, an important and widely applicable
kind of machine learning problem.
The tutorial demonstrates the basic application of transfer learning with TensorFlow Hub
and Keras.
It uses the IMDB dataset that contains the text of 50,000 movie reviews from the Internet
Movie Database. These are split into 25,000 reviews for training and 25,000 reviews for testing.
The training and testing sets are balanced, meaning they contain an equal number of positive
and negative reviews.
This script uses tf.keras, a high-level API to build and train models in TensorFlow,
and tensorflow_hub, a library for loading trained models from TFHub in a single line of code.
For a more advanced text classification tutorial using tf.keras, see the MLCC Text
Classification Guide.
---
Remember to install tensorflow-hub
$ pip install -q tensorflow-hub
$ pip install -q tensorflow-datasets
---
Original source: https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
"""

import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


def main():
    print(f'Version: {tf.__version__}')
    print(f'Eager mode: {tf.executing_eagerly()}')
    print(f'Hub version: {hub.__version__}')
    print(f'GPU is {"available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE"}')

    # Split the trainign set into 60% and 40% to end up with 15,000 examples
    # for training, 10,000 examples for validation and 25,000 examples for testing.
    train_data, validation_data, test_data = tfds.load(
        name='imdb_reviews',
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True
    )

    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    print(f'Train batch examples:\n{train_examples_batch}')
    print(f'Train batch labels:\n{train_labels_batch}')

    embedding = 'https://tfhub.dev/google/nnlm-en-dim50/2'
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
                               dtype=tf.string, trainable=True)
    print(f'{hub_layer(train_examples_batch[:3])}')
    
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    model.summary()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_data.shuffle(10000).batch(512),
        epochs=10,
        validation_data=validation_data.batch(512),
        verbose=1
    )

    #Evaluate the model
    results = model.evaluate(test_data.batch(512), verbose=2)

    for name, value in zip(model.metrics_names, results):
        print(f'{name}: {round(value, 3)}')


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
