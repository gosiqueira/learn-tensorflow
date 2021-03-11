"""
Basic text classification
---
This tutorial demonstrates text classification starting from plain text
files stored on disk. You'll train a binary classifier to perform sentiment
analysis on an IMDB dataset. At the end of the notebook, there is an exercise
for you to try, in which you'll train a multiclass classifier to predict the
tag for a programming question on Stack Overflow.
---
Original source: https://www.tensorflow.org/tutorials/keras/text_classification
"""

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )


def main():
    print(f'TensorFlow version: {tf.__version__}')

    # Sentiment analysis
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    print(f'Folders in dataset dir: {os.listdir(dataset_dir)}')

    train_dir = os.path.join(dataset_dir, 'train')
    print(f'Train folders: {os.listdir(train_dir)}')

    # Load the dataset
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    # Model training parameters
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(3):
            print(f'Review: {text_batch.numpy()[i]}')
            print(f'Label: {label_batch.numpy()[i]}')

    print(f'Label 0 corresponds to {raw_train_ds.class_names[0]}')
    print(f'Label 1 corresponds to {raw_train_ds.class_names[1]}')

    # Note: When using the validation_split and subset arguments, make sure to either
    # specify a random seed, or to pass shuffle=False, so that the validation and
    # training splits have no overlap.

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size
    )

    # Prepare the dataset for training

    # Note: to prevent train/test skew (also know as train/serving skew), it is important
    # to preprocess the data identically at train and test time. To facilitate this,
    # the TextVectorization layer can be included directly inside your model, as shown
    # later in this tutorial.
    max_features = 10000
    sequence_length = 250

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    
    # Note: it's important to only use your training data when calling adapt
    # (using the test set would leak information).
    
    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    # Retrieve a batch (of 32 reviews and labels) from the dataset
    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print(f'Review: {first_review}')
    print(f'Label: {raw_train_ds.class_names[first_label]}')
    print(f'Vectorized review: {vectorize_text(first_review, first_label)}')

    print(f'1287 ---> {vectorize_layer.get_vocabulary()[1287]}')
    print(f' 313 ---> {vectorize_layer.get_vocabulary()[313]}')
    print(f'Vocabulary size: {len(vectorize_layer.get_vocabulary())}')

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # Configure the dataset for performance

    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Create model
    embedding_dim = 16

    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.summary()

    # Loss function and optimizer
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    # Train the model
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(test_ds)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Create a plot of accuracy and loss over time
    history_dict = history.history
    print(f'History keys: {history_dict.keys()}')

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # "b" is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

    # Export the model
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss = losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )

    # Test it with 'raw_test_ds', which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds) 
    print(accuracy)

    # Inference on new data
    examples = [
        'The movie was great!',
        'The movie was okay.',
        'The movie was terrible...'
    ]

    print(f'New examples predictions: {export_model.predict(examples)}')

    # Exercise: multiclass classification on Stack Overflow questions

    # Note: to increase the difficulty of the classification problem, we have replaced any
    # occurences of the words Python, CSharp, JavaScript, or Java in the programming questions
    # with the word blank (as many questions contain the language they're about).


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
