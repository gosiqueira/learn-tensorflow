"""
Load a pandas.DataFrame
---
This tutorial provides an example of how to load pandas dataframes into a `tf.data.Dataset`.
This tutorials uses a small dataset provided by the Cleveland Clinic Foundation for Heart Disease.
There are several hundred rows in the CSV. Each row describes a patient, and each column describes
an attribute. We will use this information to predict whether a patient has heart disease,
which in this dataset is a binary classification task.
---
Original source: https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
"""

import pandas as pd
import tensorflow as tf


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def main():
    # Read data using pandas
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
    csv_file = tf.keras.utils.get_file('heart.csv', url)

    df = pd.read_csv(csv_file)
    print(df.head())
    print(f'Data types: {df.dtypes}')

    df['thal'] = pd.Categorical(df['thal'])
    df['thal'] = df.thal.cat.codes

    print(df.head())

    # Load data using `tf.data.Dataset`
    target = df.pop('target')

    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    for feat, targ in dataset.take(5):
        print(f'Features: {feat}, Target: {targ}')

    print(f'Thal values: {df["thal"]}')

    train_ds = dataset.shuffle(len(df)).batch(1)

    model = get_compiled_model()
    model.fit(train_ds, epochs=15)

    # Alternative to feature columns
    inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
    x = tf.stack(list(inputs.values()), axis=-1)

    x = tf.keras.layers.Dense(10, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)

    model_func = tf.keras.Model(inputs=inputs, outputs=output)

    model_func.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

    for dict_slice in dict_slices.take(1):
        print(f'Dictionary slice: {dict_slice}')

    model_func.fit(dict_slices, epochs=15)


if __name__ == '__main__':
    main()
