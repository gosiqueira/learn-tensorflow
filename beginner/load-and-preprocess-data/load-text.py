"""
Load text
---
This tutorial demonstrates two ways to load and preprocess text.
    - First, you will use Keras utilities and layers. If you are new to TensorFlow,
    you should start with these.
    - Next, you will use lower-level utilities like `tf.data.TextLineDataset` to load text
    files, and `tf.text` to preprocess the data for finer-grain control.
---
Remember to install TensorFlow Text
$ pip install -q -U tensorflow-text
---
Original source: https://www.tensorflow.org/tutorials/load_data/text
"""

import collections
import pathlib
import re
import string

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from tensorflow.keras import layers, losses, preprocessing, utils
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # Example 1: Predict the tag for a Stack Overflow question
    # Download and explore the dataset
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
    dataset = utils.get_file(
        'stack_overflow_16k.tar.gz',
        url,
        untar=True,
        cache_dir='stack_overflow',
        cache_subdir=''
    )
    dataset_dir = pathlib.Path(dataset).parent

    print(f'Paths: {list(dataset_dir.iterdir())}')

    train_dir = dataset_dir/'train'
    print(f'Train paths: {train_dir.iterdir}')

    sample_file = train_dir/'python/1755.txt'
    with open(sample_file) as f:
        print(f.read())

    # Load the dataset
    batch_size = 32
    seed = 42

    raw_train_ds = preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(10):
            print(f'Question: {text_batch.numpy()[i]}')
            print(f'Label: {label_batch.numpy()[i]}')

    for i, label in enumerate(raw_train_ds.class_names):
        print(f'Label {i} corresponds to {label}')

    raw_val_ds = preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )

    test_dir = dataset_dir/'test'
    raw_test_ds = preprocessing.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size
    )

    # Prepare the dataset for training
    vocab_size = 10000
    binary_vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='binary'
    )

    max_sequence_length = 250
    int_vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda text, labels: text)
    binary_vectorize_layer.adapt(train_text)
    int_vectorize_layer.adapt(train_text)

    def binary_vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return binary_vectorize_layer(text), label

    def int_vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return int_vectorize_layer(text), label

    # Retrieve a batch (of 32 reviews and labels) from the dataset
    text_batch, label_batch = next(iter(raw_train_ds))
    first_question, first_label = text_batch[0], label_batch[0]
    print(f'Question: {first_question}')
    print(f'Label: {first_label}')

    print(f'"binary" vectorized question: {binary_vectorize_text(first_question, first_label)[0]}')
    print(f'"int" vectorized question: {int_vectorize_text(first_question, first_label)[0]}')

    print(f'1298 ---> {int_vectorize_layer.get_vocabulary()[1289]}')
    print(f' 313 ---> {int_vectorize_layer.get_vocabulary()[313]}')
    print(f'Vocabulary size: {len(int_vectorize_layer.get_vocabulary())}')

    binary_train_ds = raw_train_ds.map(binary_vectorize_text)
    binary_val_ds = raw_val_ds.map(binary_vectorize_text)
    binary_test_ds = raw_test_ds.map(binary_vectorize_text)

    int_train_ds = raw_train_ds.map(int_vectorize_text)
    int_val_ds = raw_val_ds.map(int_vectorize_text)
    int_test_ds = raw_test_ds.map(int_vectorize_text)

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    def configure_dataset(dataset):
        return dataset.cache().prefetch(buffer_size=AUTOTUNE)

    binary_train_ds = configure_dataset(binary_train_ds)
    binary_val_ds = configure_dataset(binary_val_ds)
    binary_test_ds = configure_dataset(binary_test_ds)

    int_train_ds = configure_dataset(int_train_ds)
    int_val_ds = configure_dataset(int_val_ds)
    int_test_ds = configure_dataset(int_test_ds)

    # Train the model
    binary_model = tf.keras.Sequential([
        layers.Dense(4)
    ])

    binary_model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    _ = binary_model.fit(
        binary_train_ds,
        validation_data=binary_val_ds,
        epochs=10
    )

    def create_model(vocab_size, num_labels):
        model = tf.keras.Sequential([
            layers.Embedding(vocab_size, 64, mask_zero=True),
            layers.Conv1D(64, 5, padding='valid', activation='relu', strides=2),
            layers.GlobalMaxPool1D(),
            layers.Dense(num_labels)
        ])

        return model

    # vocab_size is vocab_size + 1 since 0 is used additionally for padding.
    int_model = create_model(vocab_size=vocab_size + 1, num_labels=4)
    int_model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    _ = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=5)

    print(f'Linear model on binary vectorized data: {binary_model.summary()}')

    print(f'ConvNet model on int vectorized data: {int_model.summary()}')

    _, binary_accuracy = binary_model.evaluate(binary_test_ds)
    _, int_accuracy = int_model.evaluate(int_test_ds)

    print(f'Binary model accuracy {binary_accuracy:2.2%}')
    print(f'Int model accuracy: {int_accuracy:2.2%}')

    # Export the model
    export_model = tf.keras.Sequential([
        binary_vectorize_layer,
        binary_model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(f'Accuracy: {accuracy:2.2%}')

    def get_string_labels(predicted_scores_batch):
        predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
        predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
        return predicted_labels

    # Run inference on new data
    inputs = [
        'how do I extract keys from a dict into a list?',       # python
        'debug public static void main(string[] args) {...}',   # java
    ]

    predicted_scores = export_model.predict(inputs)
    predicted_labels = get_string_labels(predicted_scores)
    for input, label in zip(inputs, predicted_labels):
        print(f'Question: {input}')
        print(f'Predicted label: {label.numpy()}')

    # Example 2: Predict the author of Illiad translations
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
    file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

    for name in file_names:
        text_dir = utils.get_file(name, origin=url + name)

    parent_dir = pathlib.Path(text_dir).parent
    print(f'Paths: {list(parent_dir.iterdir())}')

    # Load the dataset
    def labeler(example, index):
        return example, tf.cast(index, tf.int64)

    labeled_data_sets = []

    for i, file_name in enumerate(file_names):
        lines_dataset = tf.data.TextLineDataset(str(parent_dir/file_name))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)

    buffer_size = 50000
    batch_size = 64
    validation_size = 5000

    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(buffer_size, reshuffle_each_iteration=False)

    for text, label in all_labeled_data.take(10):
        print(f'Sentence: {text.numpy()}')
        print(f'Label: {label.numpy()}')

    # Prepare the dataset for training
    tokenizer = tf_text.UnicodeScriptTokenizer()

    def tokenize(text, unused_label):
        lower_case = tf_text.case_fold_utf8(text)
        return tokenizer.tokenize(lower_case)

    tokenized_ds = all_labeled_data.map(tokenize)

    for text_batch in tokenized_ds.take(5):
        print(f'Tokens: {text_batch.numpy()}')

    tokenized_ds = configure_dataset(tokenized_ds)

    vocab_dict = collections.defaultdict(lambda: 0)
    for toks in tokenized_ds.as_numpy_iterator():
        for tok in toks:
            vocab_dict[tok] += 1

    vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    vocab = [token for token, count in vocab]
    vocab = vocab[:vocab_size]
    vocab_size = len(vocab)

    print(f'Vocab size: {vocab_size}')
    print(f'First five vocab entries: {vocab[:5]}')

    keys = vocab
    values = range(2, len(vocab) + 2)  # reserve 0 for padding, 1 for OOV

    init = tf.lookup.KeyValueTensorInitializer(
        keys, values, key_dtype=tf.string, value_dtype=tf.int64
    )

    num_oov_buckets = 1
    vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)

    def preprocess_text(text, label):
        standardized = tf_text.case_fold_utf8(text)
        tokenized = tokenizer.tokenize(standardized)
        vectorized = vocab_table.lookup(tokenized)
        return vectorized, label

    example_text, example_label = next(iter(all_labeled_data))
    print(f'Sentence: {example_text.numpy()}')
    vectorized_text, example_label = preprocess_text(example_text, example_label)
    print(f'Vectorized sentence: {vectorized_text.numpy()}')

    all_encoded_data = all_labeled_data.map(preprocess_text)

    # Split the dataset into train and test
    train_data = all_encoded_data.skip(validation_size).shuffle(buffer_size)
    validation_data = all_encoded_data.take(validation_size)

    train_data = train_data.padded_batch(batch_size)
    validation_data = validation_data.padded_batch(batch_size)

    sample_text, sample_labels = next(iter(validation_data))
    print(f'Text batch shape: {sample_text.shape}')
    print(f'Label batch shape: {sample_labels.shape}')
    print(f'First text example: {sample_text[0]}')
    print(f'First label example: {sample_labels[0]}')

    vocab_size += 2

    # Train the model
    model = create_model(vocab_size=vocab_size, num_labels=3)
    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    _ = model.fit(train_data, validation_data=validation_data, epochs=3)

    loss, accuracy = model.evaluate(validation_data)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy:2.2%}')

    # Export the model
    preprocess_layer = TextVectorization(
        max_tokens=vocab_size,
        standardize=tf_text.case_fold_utf8,
        split=tokenizer.tokenize,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )

    preprocess_layer.set_vocabulary(vocab)

    export_model = tf.keras.Sequential([
        preprocess_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Create a test dataset of raw strings
    test_ds = all_labeled_data.take(validation_size).batch(batch_size)
    test_ds = configure_dataset(test_ds)
    loss, accuracy = export_model.evaluate(test_ds)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy:2.2%}')

    # Run inference on new data
    inputs = [
        "Join'd to th' Ionians with their flowing robes,",                          # Label: 1
        "the allies, and his armour flashed about him so that he seemed to all",    # Label: 2
        "And with loud clangor of his arms he fell.",                               # Label: 0
    ]
    predicted_scores = export_model.predict(inputs)
    predicted_labels = tf.argmax(predicted_scores, axis=1)
    for input, label in zip(inputs, predicted_labels):
        print(f'Question: {input}')
        print(f'Predicted label: {label.numpy()}')

    # Download more datasets using TensorFlow Datasets (TFDS)
    train_ds = tfds.load(
        'imdb_reviews',
        split='train',
        batch_size=batch_size,
        shuffle_files=True,
        as_supervised=True
    )

    val_ds = tfds.load(
        'imdb_reviews',
        split='train',
        batch_size=batch_size,
        shuffle_files=True,
        as_supervised=True
    )

    for review_batch, label_batch in val_ds.take(1):
        for i in range(5):
            print(f'Review: {review_batch[i].numpy()}')
            print(f'Label: {label_batch[i].numpy()}')

    # Prepare thje dataset for training
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = train_ds.map(lambda text, labels: text)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = train_ds.map(vectorize_text)
    val_ds = val_ds.map(vectorize_text)

    train_ds = configure_dataset(train_ds)
    val_ds = configure_dataset(val_ds)

    # Train the model
    model = create_model(vocab_size=vocab_size + 1, num_labels=1)
    model.summary()

    model.compile(
        optimizer='adam',
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    _ = model.fit(train_ds, validation_data=val_ds, epochs=3)

    loss, accuracy = model.evaluate(val_ds)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy:2.2%}')

    # Export the model
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 0 ---> negative review
    # 1 ---> positive review
    inputs = [
        'This is a fantastic movie.',
        'This is a bad movie.',
        'This movie was bad that it was good.',
        'I will never say yes to watching this movie.'
    ]
    predicted_scores = export_model.predict(inputs)
    predicted_labels = [int(round(x[0])) for x in predicted_scores]
    for input, label in zip(inputs, predicted_labels):
        print(f'Question: {input}')
        print(f'Predicted label: {label}')

    
if __name__ == '__main__':
    main()
