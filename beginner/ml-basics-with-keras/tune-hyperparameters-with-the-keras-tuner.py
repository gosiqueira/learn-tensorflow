"""
Introduction to the Keras Tuner
---
Overview
The Keras Tuner is a library that helps you pick the optimal set of hyperparameters for
your TensorFlow program. The process of selecting the right set of hyperparameters for
your machine learning (ML) application is called hyperparameter tuning or hypertuning.
Hyperparameters are the variables that govern the training process and the topology
of an ML model. These variables remain constant over the training process and directly
impact the performance of your ML program. Hyperparameters are of two types:
    1. Model hyperparameters which influence model selection such as the number and width
    of hidden layers
    2. Algorithm hyperparameters which influence the speed and quality of the learning
    algorithm such as the learning rate for Stochastic Gradient Descent (SGD) and the number
    of nearest neighbors for a k Nearest Neighbors (KNN) classifier
In this tutorial, you will use the Keras Tuner to perform hypertuning for an image classification
application.
---
Remember to install Keras Tuner
$ pip install -q -U keras-tuner
---
Original source: https://www.tensorflow.org/tutorials/keras/keras_tuner
"""

import kerastuner as kt
import tensorflow as tf

from tensorflow import keras


def model_builder(hp):
    # Define the model
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal values between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def main():
    # Download and prepare the dataset
    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values batween 0 and 1
    img_train = img_train.astype('float32') / 255.0
    img_test = img_test.astype('float32') / 255.0

    # Intantiate the tuner and perform hypertuning
    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='my_dir',
        project_name='intro_to_kt'
    )

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trails=1)[0]

    print(f"""
        The hyperparameter search is complete. The optimal number of units in the first
        densely-connected layer is {best_hps.get('units')} and the optimal learning rate
        for the optimizer is {best_hps.get('learning_rate')}.
        """
    )

    # Train the model

    # Build the model with the optimal hyperparameters and train it on the data
    # for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        img_train,
        label_train,
        epochs=50,
        validation_split=0.2,
    )

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print(f'Best epoch: {best_epoch}')

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.hit(img_test, label_test, epoch=best_epoch)

    eval_result = hypermodel.evaluate(img_test, label_test)
    print(f'[test loss, test accuracy]: {eval_result}')


if __name__ == '__main__':
    main()