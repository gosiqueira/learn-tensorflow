"""
Load Images
---
This tutorial shows how to load and preprocess an image dataset in three ways. First,
you will use high-level Keras preprocessing utilities and layers to read a directory
of images on disk. Next, you will write your own input pipeline from scratch using tf.data.
Finally, you will download a dataset from the large catalog available in TensorFlow Datasets.
---
Original source: https://www.tensorflow.org/tutorials/load_data/images
"""

import os
import pathlib

import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

from matplotlib import pyplot as plt
from tensorflow.keras import layers


def main():
    print(f'TensorFlow Version: {tf.__version__}')

    # Download the flowers dataset
    url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    data_dir = tf.keras.utils.get_file(
        origin=url,
        fname='flower_photos',
        untar=True
    )

    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f'Number of images: {image_count}')

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))
    PIL.Image.open(str(roses[1]))

    # Load using keras.preprocessing
    # Note: The Keras Preprocesing utilities and layers introduced in this section
    # are currently experimental and may change.

    # Create a dataset
    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f'Class names: {class_names}')

    # Visualize the data
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            _ = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis('off')

    # Standardize the data
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, _ = next(iter(normalized_ds))
    first_image = image_batch[0]

    # Notice the pixels values are now in [0, 1]
    print(f'Pixels values interval: [{np.min(first_image)}, {np.max(first_image)}]')

    # Note: If you would like to scale pixel values to [-1,1] you can instead write
    # Rescaling(1./127.5, offset=-1)

    # Note: we previously resized images using the image_size argument of
    # image_dataset_from_directory. If you want to include the resizing logic
    # in your model, you can use the Resizing layer instead.

    # Configure the dataste for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Train a model
    num_classes = 5

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Note: we will only train for a few epochs so this tutorial runs quickly.
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    # Using tf.data for finer control
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*'/'*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    for f in list_ds.take(1):
        print(f.numpy())

    class_names = np.array(sorted([
        item.name for item in data_dir.glob('*') if item.name != 'LICENSE.txt'
    ]))
    print('Class names: {class_names}')

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print(f'Train data cardinality: {tf.data.experimental.cardinality(train_ds).numpy()}')
    print(f'Val data cardinality: {tf.data.experimental.cardinality(val_ds).numpy()}')

    def get_label(file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)


    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])


    def process_path(file_path):
        label = get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set num_parallel_calls so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image, labels in train_ds.take(1):
        print(f'Image shape: {image.numpy().shape}')
        print(f'Label: {labels.numpy()}')

    # Configure dataset for performance
    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    # Visualize the data
    image_batch, label_batch = next(iter(train_ds))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        _ = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        label = label_batch[i]
        plt.title(class_names[label])
        plt.axis('off')

    # Continue training the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    # Using TensorFlow Datasets
    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    num_classes = metadata.features['label'].num_classes
    print(f'Number of classes: {num_classes}')

    get_label_name = metadata.features['label'].int2str

    image, label = next(iter(train_ds))
    _ = plt.imshow(image)
    _ = plt.title(get_label_name(label))

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)


if __name__ == '__main__':
    main()
