"""
Custom Training: Walkthrough
---
This guide uses machine learnign to categorize Iris flowers by species. It uses TensorFlow to:
    1. Build a model,
    2. Train this model on example data, and
    3. Use the model to make predictions about unknown data.

TensorFlow programming

This guide uses these high-level TensorFlow concepts:
    - Use TensorFlow's default eager execution development environment,
    - Import data with the Datasets API,
    - Build models and layers with TensorFlow's Keras API.
This tutorial is structured like many TensorFlow programs:
    1. Import and parse the dataset.
    2. Select the type of model.
    3. Train the model.
    4. Evaluate the model's effectiveness.
    5. Use the trained model to make predictions.

Setup program

Configure imports

Import TensorFlow and the other required Python modules. By default, TensorFlow uses eager
execution to evaluate operations immediately, returning concrete values instead of creating
a computational graph that is executed later. If you are used to a REPL or the python
interactive console, this feels familiar.
---
Original source: https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

import os

import matplotlib.pyplot as plt
import tensorflow as tf


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def main():
    print(f'TensorFlow version: {tf.__version__}')
    print(f'Eager execution: {tf.executing_eagerly}')

    # The Iris classification problem
    train_dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'
    train_dataset_fp = tf.keras.utils.get_file(
        fname=os.path.basename(train_dataset_url),
        origin=train_dataset_url
    )
    print(f'Local copy of the dataset file: {train_dataset_fp}')

    # Column order in CSV file
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    feature_names = column_names[:-1]
    label_name = column_names[-1]
    
    print(f'Features: {feature_names}')
    print(f'Label: {label_name}')

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

    batch_size = 32
    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1
    )

    features, labels = next(iter(train_dataset))
    print(features)

    plt.scatter(features['petal_length'], features['sepal_length'], c=labels, cmap='viridis')
    plt.xlabel('Petal length')
    plt.ylabel('Sepal length')
    plt.show()

    train_dataset = train_dataset.map(pack_features_vector)

    features, labels = next(iter(train_dataset))
    print(features[:5])

    # Select the type of model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    # Using the model
    predictions = model(features)
    print(f'Predictions: {predictions[:5]}')
    print(f'Softmax predictions: {tf.nn.softmax(predictions[:5])}')

    print(f'Predictions: {tf.argmax(predictions, axis=1)}')
    print(f'     Labels: {labels}')

    # Train the model
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def loss(model, x, y, training):
        """
        training=training is needed only if there are layers with different behavior during training
        versus inference (e.g. Dropout).
        """
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

    l = loss(model, features, labels, training=False)
    print(f'Loss test: {l}')

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    loss_value, grads = grad(model, features, labels)
    print(f'Step {optimizer.iterations.numpy()}, Initial Loss: {loss_value.numpy()}')
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'Step {optimizer.iterations.numpy()},         Loss: {loss(model, features, labels, training=True).numpy()}')

    # Traning loop
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value) # Add current batch loss
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}: Loss: {epoch_loss_avg.result():.3f}, Accuracy: {epoch_accuracy.result():.3f}')

    # Visualize the loss fucntion over time
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel('Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].plot(train_accuracy_results)

    plt.show()

    # Evaluate the model effectiveness
    # Setup the test dataset
    test_url = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'
    test_dataset_fp = tf.keras.utils.get_file(
        fname=os.path.basename(test_url),
        origin=test_url
    )

    test_dataset = tf.data.experimental.make_csv_dataset(
        test_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name='species',
        num_epochs=1,
        shuffle=False
    )

    test_dataset = test_dataset.map(pack_features_vector)

    # Evaluate the model on the test dataset
    test_accuracy = tf.keras.metrics.Accuracy()

    for x, y in test_dataset:
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print(f'Test set accuracy: {test_accuracy.result():.3f}')

    print(tf.stack([y, prediction], axis=1))

    # Use the trained model to make predictions
    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])

    predictions = model(predict_dataset, training=False)

    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)[class_idx]
        name = class_names[class_idx]
        print(f'Example {i} prediction: {name} ({100*p:4.1f})')


if __name__ == '__main__':
    main()
