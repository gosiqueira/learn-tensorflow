"""
Basic regression: Predict fuel efficiency
---
In a regression problem, the aim is to predict the output of a continuous value,
like a price or a probability. Contrast this with a classification problem, where
the aim is to select a class from a list of classes (for example, where a picture
contains an apple or an orange, recognizing which fruit is in the picture).
This script uses the classic Auto MPG Dataset and builds a model to predict the
fuel efficiency of late-1970s and early 1980s automobiles. To do this, provide
the model with a description of many automobiles from that time period. This description
includes attributes like: cylinders, displacement, horsepower, and weight.
This example uses the tf.keras API, see this guide for details.
---
Remember to install seaborn
$ pip install -q seaborn
---
Original source: https://www.tensorflow.org/tutorials/keras/regression
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make numpy printous easier to read
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_horsepower(x, y, train_features, train_labels):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()

    plt.show()


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='mean_absolute_error'
    )

    return model


def main():
    print(f'TensorFlow version: {tf.__version__}')
    
    # The Auto MPG dataset
    # Get the data

    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = [
        'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
        'Acceleration', 'Model Year', 'Origin'
    ]

    raw_dataset = pd.read_csv(
        url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True
    )
    dataset = raw_dataset.copy()
    print(dataset.tail())

    # Clean the data
    print(f'Dataset NA count: {dataset.isna().sum()}')
    
    dataset = dataset.dropna()

    # Note: You can set up the keras.Model to do this kind of transformation for you.
    # That's beyond the scope of this tutorial. See the preprocessing layers or
    # Loading CSV data tutorials for examples.

    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    print(dataset.tail())

    # Split the data into train and test
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Inspect the data
    sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

    print(train_dataset.describe().transpose())

    # Split features from labels
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    # Normalization
    print(train_dataset.describe().transpose()[['mean', 'std']])

    # The Normalization layer
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))

    print(f'Columns mean value: {normalizer.mean.numpy()}')

    first = np.array(train_features[:1])
    with np.printoptions(precision=2, suppress=True):
        print(f'First example: {first}')
        print()
        print(f'First example normalized: {normalizer(first).numpy()}')

    # Linear regression

    # One variable
    horsepower = np.array(train_features['Horsepower'])

    horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
    horsepower_normalizer.adapt(horsepower)

    horsepower_model = tf.keras.Sequential([
        horsepower_normalizer,
        layers.Dense(1)
    ])

    print(horsepower_model.summary())

    print(f'Horsepower model preds: {horsepower_model.predict(horsepower[:10])}')
    
    horsepower_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )

    history = horsepower_model.fit(
        train_features['Horsepower'], train_labels,
        epochs=100,
        # Suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    plot_loss(history)

    test_results = {}

    test_results['horsepower_model'] = horsepower_model.evaluate(
        test_features['Horsepower'],
        test_labels,
        verbose=0
    )

    x = tf.linspace(0.0, 250, 251)
    y = horsepower_model.predict(x)

    plot_horsepower(x, y, train_features, train_labels)

    # Multiple inputs
    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(1)
    ])

    print(f'Linear model preds: {linear_model.predict(train_features[:10])}')
    print(f'Kernel weights: {linear_model.layers[1].kernel}')

    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )

    history = linear_model.fit(
        train_features,
        train_labels,
        epochs=100,
        # Suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2
    )

    plot_loss(history)

    test_results['linear_model'] = linear_model.evaluate(
        test_features,
        test_labels,
        verbose=0
    )

    # A DNN regression
    # One variable
    dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
    print(dnn_horsepower_model.summary())

    history = dnn_horsepower_model.fit(
        train_features['Horsepower'],
        train_labels,
        validation_split=0.2,
        verbose=0,
        epochs=100
    )

    plot_loss(history)

    x = tf.linspace(0.0, 250, 251)
    y = dnn_horsepower_model.predict(x)

    plot_horsepower(x, y, train_features, train_labels)

    test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
        test_features['Horsepower'],
        test_labels,
        verbose=0
    )

    # Full model
    dnn_model = build_and_compile_model(normalizer)
    print(dnn_model.summary())

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0,
        epochs=100
    )

    plot_loss(history)

    test_results['dnn_model'] = dnn_model.evaluate(
        test_features,
        test_labels,
        verbose=0
    )

    # Performance
    print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

    # Make predicitons
    test_predictions = dnn_model.predict(test_features).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [MPG]')
    _ = plt.ylabel('Count')

    dnn_model.save('dnn_model')

    reloaded = tf.keras.models.load_model('dnn_model')
    test_results['reloaded'] = reloaded.evaluate(
        test_features,
        test_labels,
        verbose=0
    )

    print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)


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
