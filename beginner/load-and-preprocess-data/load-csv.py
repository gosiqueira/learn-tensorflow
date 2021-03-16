"""
Load CSV data
---
This tutorial provides examples of how to use CSV data with TensorFlow.
There are two main parts to this:
    - Loading the data off disk
    - Pre-processing it into a form suitable for training.
This tutorial focuses on the loading, and gives some quick examples of preprocessing.
For a tutorial that focuses on the preprocessing aspect see the preprocessing layers
guide and tutorial.
---
Original source: https://www.tensorflow.org/tutorials/load_data/csv
"""

import itertools
import pathlib
import re
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


def get_titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])

    preprocessing_inputs = preprocessing_head(inputs)
    result = body(preprocessing_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.optimizers.Adam())

    return model


def slices(features):
    for i in itertools.count():
        # For each feature take index `i`
        example = {name: values[i] for name, values in features.items()}
        yield example


def make_images(features):
    image = [None] * 400
    new_feats = {}

    for name, value in features.items():
        match = re.match('r(\\d+)c(\\d+)', name)
        if match:
            image[int(match.group(1)) * 20 + int(match.group(2))] = value
        else:
            new_feats[name] = value

    image = tf.stack(image, axis=0)
    image = tf.reshape(image, [20, 20, -1])
    new_feats['image'] = image

    return new_feats


def main():
    # In memory data
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv'
    abalone_train = pd.read_csv(url, names=[
        'Length', 'Diamenter', 'Height', 'Whole weight', 'Viscera weight', 'Shell weight', 'Age'
    ])

    print(abalone_train.head())

    abalone_features = abalone_train.copy()
    abalone_labels = abalone_features.pop('Age')

    abalone_features = np.array(abalone_features)
    print(f'Features: {abalone_features}')

    abalone_model = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])

    abalone_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

    # Basic preprocessing
    normalize = preprocessing.Normalization()

    normalize.adapt(abalone_features)

    norm_abalone_model = tf.keras.Sequential([
        normalize,
        layers.Dense(64),
        layers.Dense(1)
    ])

    norm_abalone_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
    norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)

    # Mixed data types
    url = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
    titanic = pd.read_csv(url)
    print(titanic.head())

    titanic_features = titanic.copy()
    titanic_labels = titanic_features.pop('survived')

    # Create a symbolic input
    input = tf.keras.Input(shape=(), dtype=tf.float32)

    # Do a calculation using is
    result = 2 * input + 1

    # The result doesn't have a value
    print(f'Result: {result}')

    calc = tf.keras.Model(inputs=input, outputs=result)

    print(f'calc(1) = {calc(1).numpy()}')
    print(f'calc(2) = {calc(2).numpy()}')

    inputs = {}
    for name, column in titanic_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    inputs

    numeric_inputs = {name: input for name, input in inputs.items() if input.dtype == tf.float32}

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = preprocessing.Normalization()
    norm.adapt(np.array(titanic[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)

    all_numeric_inputs

    preprocessed_inputs = [all_numeric_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        lookup = preprocessing.StringLookup(vocabulary=np.unique(titanic_features[name]))
        one_hot  = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

    titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    tf.keras.utils.plot_model(model=titanic_preprocessing, rankdir='LR', dpi=72, show_shapes=True)

    titanic_features_dict = {name: np.array(value) for name, value in titanic_features.items()}

    features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}
    titanic_preprocessing(features_dict)

    titanic_model = get_titanic_model(titanic_preprocessing, inputs)

    titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

    titanic_model.save('test')
    reloaded = tf.keras.models.load_model('test')

    features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}

    before = titanic_model(features_dict)
    after = reloaded(features_dict)
    assert (before - after) < 1e-3
    print(f'Before: {before}')
    print(f'After: {after}')

    # Using tf.data
    # On in memory datasets
    for example in slices(titanic_features_dict):
        for name, value in example.items():
            print(f'{name:19s}: {value}')
        break

    titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))

    titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)

    titanic_model.fit(titanic_batches, epochs=5)

    # From a single file
    url = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
    titanic_file_path = tf.keras.utils.get_file('train.csv', url)

    titanic_csv_ds = tf.data.experimental.make_csv_dataset(
        titanic_file_path,
        batch_size=5,           # Artificiallly small to make examples easier to show.
        label_name='survived',
        num_epochs=1,
        ignore_errors=True,
    )

    for batch, label in titanic_csv_ds.take(1):
        for key, value in batch.items():
            print(f'{key:20s}: value')
        print()
        print(f'{"label":20s}: {label}')

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'
    traffic_volume_csv_gz = tf.keras.utils.get_file(
        'Metro_Interstate_Traffic_Volume.csv.gz',
        url,
        cache_dir='.',
        cache_subdir='traffic'
    )

    traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
        traffic_volume_csv_gz,
        batch_size=256,
        label_name='traffic_volume',
        num_epochs=1,
        compression_type='GZIP'
    )

    for batch, label in traffic_volume_csv_gz_ds.take(1):
        for key, value in batch.items():
            print(f'{key:20s}: {value[:5]}')
        print()
        print(f'{"label":20s}: {label[:5]}')

    #Caching
    start = time.time()
    for i, (batch, label) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):
        if i % 40 == 0:
            print('.', end='')
    print(f'Total time: {time.time() - start:.3f}')

    caching = traffic_volume_csv_gz_ds.cache().shuffle(1000)

    start = time.time()
    for i, (batch, label) in enumerate(caching.shuffle(1000).repeat(20)):
        if i % 40 == 0:
            print('.', end='')
    print(f'Total time: {time.time() - start:.3f}')

    start = time.time()
    snapshot = tf.data.experimental.snapshot('titanic.tfsnap')
    snapshotting = traffic_volume_csv_gz_ds.apply(snapshot).shuffle(1000)

    for i, (batch, label) in enumerate(snapshotting.shuffle(1000).repeat(20)):
        if i % 40 == 0:
            print('.', end='')
    print(f'Total time: {time.time() - start:.3f}')

    # Multiple files
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip'
    _ = tf.keras.utils.get_file(
        'fonts.zip',
        url,
        cache_dir='.',
        cache_subdir='fonts',
        extract=True
    )

    fonts_csvs = sorted(str(p) for p in pathlib.Path('fonts').glob('*.csv'))

    print(f'Fonts: {fonts_csvs[:10]}')
    print(f'Fonts len: {len(fonts_csvs)}')

    fonts_ds = tf.data.experimental.make_csv_dataset(
        file_pattern='fonts/*.csv',
        batch_size=10,
        num_epochs=1,
        num_parallel_reads=20,
        shuffle_buffer_size=10000
    )

    for features in fonts_ds.take(1):
        for i, (name, value) in enumerate(features.items()):
            if i > 15:
                break
            print(f'{name:20s}: {value}')
    print('...')
    print(f'[total: {len(features)} features]')

    # Optional: Packing fields
    fonts_image_ds = fonts_ds.map(make_images)

    for features in fonts_image_ds.take(1):
        break

    plt.figure(figsize=(6, 6), dpi=120)

    for n in range(9):
        plt.subplot(3, 3, n+1)
        plt.imshow(features['image'][..., n])
        plt.title(chr(features['m_label'][n]))
        plt.axis('off')

    plt.show()

    # Lower level functions
    # `tf.io.decode_csv`
    text = pathlib.Path(titanic_file_path).read_text()
    lines = text.split('\n')[1:-1]

    all_strings = [str()] * 10
    print(f'{all_strings}')

    features = tf.io.decode_csv(lines, record_defaults=all_strings)

    for f in features:
        print(f'type: {f.dtype.name}, shape: {f.shape}')

    print(f'Sample record: {lines[0]}')

    titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
    print(f'Data types: {titanic_types}')

    features = tf.io.decode_csv(lines, record_defaults=titanic_types)

    for f in features:
        print(f'type: {f.dtype.name}, shape: {f.shape}')

    # `tf.data.experimental.CsvDataset`
    simple_titanic = tf.data.experimental.CsvDataset(
        titanic_file_path,
        record_defaults=titanic_types,
        header=True
    )

    for example in simple_titanic.take(1):
        print(f'Sample record: {[e.numpy() for e in example]}')

    
    def decode_titanic_line(line):
        return tf.io.decode_csv(line, titanic_types)

    manual_titanic = (
        # Load the lines of text
        tf.data.TextLineDataset(titanic_file_path)
        # Skip the header row
        .skip(1)
        # Decode the line
        .map(decode_titanic_line)
    )

    for example in manual_titanic.take(1):
        print(f'Sample record: {[e.numpy() for e in example]}')

    # Multiple files
    font_line = pathlib.Path(fonts_csvs[0]).read_text().splitlines()[1]
    print(f'Sample: {font_line}')

    num_font_features = font_line.count(',') + 1
    font_column_types = [str(), str()] + [float()] * (num_font_features - 2)

    print(f'Fonts[0]: {fonts_csvs[0]}')

    simple_font_ds = tf.data.experimental.CsvDataset(
        fonts_csvs,
        record_defaults=font_column_types,
        header=True
    )

    for row in simple_font_ds.take(10):
        print(f'CSV first column: {row[0].numpy()}')

    font_files = tf.data.Dataset.list_files('fonts/*.csv')

    print('Epoch 1:')
    for f in list(font_files)[:5]:
        print(f'    {f.numpy()}')
    print('    ...')
    print()
    print('Epoch 2:')
    for f in list(font_files)[:5]:
        print(f'    {f.numpy()}')
    print('    ...')

    def make_font_csv_ds(path):
        return tf.data.experimental.CsvDataset(
            path,
            record_defaults=font_column_types,
            header=True
        )

    font_rows = font_files.interleave(make_font_csv_ds, cycle_length=3)

    fonts_dict = {'font_name': [], 'character': []}

    for row in font_rows.take(10):
        fonts_dict['font_name'].append(row[0].numpy().decode())
        fonts_dict['character'].append(chr(row[2].numpy()))

    print(pd.DataFrame(fonts_dict))

    # Performance
    BATCH_SIZE=2048
    font_ds = tf.data.experimental.make_csv_dataset(
        file_pattern='fonts/*.csv',
        batch_size=BATCH_SIZE,
        num_epochs=1,
        num_parallel_reads=100
    )

    start = time.time()
    for i, batch in enumerate(font_ds.take(20)):
        print('.', end='')
    print(f'Total time: {time.time() - start:.3f}')


if __name__ == '__main__':
    main()
