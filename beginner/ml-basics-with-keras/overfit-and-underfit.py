"""
Overfit and underfit
---
As always, the code in this example will use the tf.keras API, which you can learn
more about in the TensorFlow Keras guide.
In both of the previous examples—classifying text and predicting fuel
efficiency — we saw that the accuracy of our model on the validation
data would peak after training for a number of epochs, and would then
stagnate or start decreasing.
In other words, our model would overfit to the training data. Learning how
to deal with overfitting is important. Although it's often possible to achieve
high accuracy on the training set, what we really want is to develop models that
generalize well to a testing set (or data they haven't seen before).
The opposite of overfitting is underfitting. Underfitting occurs when there is still
room for improvement on the train data. This can happen for a number of reasons:
If the model is not powerful enough, is over-regularized, or has simply not been trained
long enough. This means the network has not learned the relevant patterns in the training data.
If you train for too long though, the model will start to overfit and learn patterns from
the training data that don't generalize to the test data. We need to strike a balance.
Understanding how to train for an appropriate number of epochs as we'll explore below is
a useful skill.
To prevent overfitting, the best solution is to use more complete training data. The dataset
should cover the full range of inputs that the model is expected to handle. Additional data
may only be useful if it covers new and interesting cases.
A model trained on more complete data will naturally generalize better. When that is no
longer possible, the next best solution is to use techniques like regularization. These
place constraints on the quantity and type of information your model can store. If a network
can only afford to memorize a small number of patterns, the optimization process will force it
to focus on the most prominent patterns, which have a better chance of generalizing well.
In this script, we'll explore several common regularization techniques, and use them to improve
on a classification model.
---
Remember to install tensorflow-docs
$ pip install -q git+https://github.com/tensorflow/docs
---
Original source: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
"""

import pathlib
import shutil
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from matplotlib import pyplot as plt
from tensorflow.keras import layers, regularizers

FEATURES = 28
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE


def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label


def get_optimizer():
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        1e-3,
        decay_steps=STEPS_PER_EPOCH * 1000,
        decay_rate=1,
        staircase=False
    )

    return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks(logdir, name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name)
    ]


def compile_and_fit(model, logdir, name, train_ds, validation_ds, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
            'accuracy'
        ]
    )

    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validation_ds,
        callbacks=get_callbacks(logdir, name),
        verbose=0
    )

    return history


def main():
    print(f'TensorFlow version: {tf.__version__}')

    logdir = pathlib.Path(tempfile.mkdtemp())/'tensorboard_logs'
    shutil.rmtree(logdir, ignore_errors=True)

    # The Higgs Dataset
    url = 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz'
    gz = tf.keras.utils.get_file('HIGGS.csv.gz', url)

    ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type='GZIP')

    packed_ds = ds.batch(10000).map(pack_row).unbatch()

    for features, _ in packed_ds.batch(1000).take(1):
        print(f'Features: {features[0]}')
        plt.hist(features.numpy().flatten(), bins=101)

    validation_ds = packed_ds.take(N_VALIDATION).cache()
    train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

    print(f'Train dataset: {train_ds}')

    validation_ds = validation_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

    # Demonstrate overfitting
    # Training procedure
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        1e-3,
        decay_steps=STEPS_PER_EPOCH * 1000,
        decay_rate=1,
        staircase=False
    )

    step = np.linspace(0, 100000)
    lr = lr_schedule(step)
    plt.figure(figsize=(8, 6))
    plt.plot(step / STEPS_PER_EPOCH, lr)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.show()

    # Tiny model
    tiny_model = tf.keras.Sequential([
        layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
        layers.Dense(1)
    ])

    size_histories = {}

    size_histories['Tiny'] = compile_and_fit(
        tiny_model,
        logdir,
        'sizes/Tiny',
        train_ds,
        validation_ds
    )

    plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
    plotter.plot(size_histories)
    plt.ylim([0.5, 0.7])

    plt.show()

    # Small model
    small_model = tf.keras.Sequential([
        layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
        layers.Dense(16, activation='elu'),
        layers.Dense(1)
    ])

    size_histories['Small'] = compile_and_fit(
        small_model,
        logdir,
        'sizes/Small',
        train_ds,
        validation_ds
    )

    # Medium model
    medium_model = tf.keras.Sequential([
        layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
        layers.Dense(64, activation='elu'),
        layers.Dense(64, activation='elu'),
        layers.Dense(1)
    ])

    size_histories['Medium'] = compile_and_fit(
        medium_model,
        logdir,
        'sizes/Medium',
        train_ds,
        validation_ds
    )

    # Large model
    large_model = tf.keras.Sequential([
        layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
        layers.Dense(512, activation='elu'),
        layers.Dense(512, activation='elu'),
        layers.Dense(512, activation='elu'),
        layers.Dense(1)
    ])

    size_histories['Large'] = compile_and_fit(
        large_model,
        logdir,
        'sizes/Large',
        train_ds,
        validation_ds
    )

    # Plot the trainign and validation losses
    plotter.plot(size_histories)
    a = plt.xscale('log')
    plt.xlim([5, max(plt.xlim())])
    plt.ylim([0.5, 0.7])
    plt.xlabel('Epochs [Log Scale]')

    plt.show()

    # Note: All the above training runs used the callbacks.EarlyStopping to end the training
    # once it was clear the model was not making progress.

    # View in TensorBoard
    # $ tensorboard --logdir {logdir}/sizes

    # If you want to share TensorBoard results you can upload the logs to TensorBoard.dev
    # by copying the following command.
    # Note: This step requires a Google account.
    # $ tensorboard dev upload --logdir {logdir}/sizes

    # Strategies to prevent oiverfitting
    shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
    shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

    regularizer_histories = {}
    regularizer_histories['Tiny'] = size_histories['Tiny']

    # Add weight regularization
    l2_model = tf.keras.Sequential([
        layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(1e-3), input_shape=(FEATURES,)),
        layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dense(1)
    ])

    regularizer_histories['l2'] = compile_and_fit(
        l2_model,
        logdir,
        'regularizers/l2',
        train_ds,
        validation_ds
    )

    plotter.plot(regularizer_histories)
    plt.ylim([0.5, 0.7])

    plt.show()

    # Add dropout
    dropout_model = tf.keras.Sequential([
        layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
        layers.Dropout(0.5),
        layers.Dense(512, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])

    regularizer_histories['dropout'] = compile_and_fit(
        dropout_model,
        logdir,
        'regularizers/dropout',
        train_ds,
        validation_ds
    )

    plotter.plot(regularizer_histories)
    plt.ylim([0.5, 0.7])

    plt.show()

    # Combine L2 + dropout
    combined_model = tf.keras.Sequential([
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                    activation='elu', input_shape=(FEATURES,)),
        layers.Dropout(0.5),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                    activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                    activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                    activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])

    regularizer_histories['combined'] = compile_and_fit(
        combined_model,
        logdir,
        'regularizers/combined',
        train_ds,
        validation_ds
    )

    plotter.plot(regularizer_histories)
    plt.ylim([0.5, 0.7])

    plt.show()

    # View in TensorBoard
    # $ tensorboard --logdir {logdir}/regularizers
    # $ tensorboard dev upload --logdir  {logdir}/regularizers


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
