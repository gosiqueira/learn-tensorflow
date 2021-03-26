"""
Customization basics: tensors and operations
---
This is an introductory TensorFlow tutorial that shows how to:
    - Import the required package
    - Create and use tensors
    - Use GPU acceleration
    - Demonstrate `tf.data.Dataset`
---
Original source: https://www.tensorflow.org/tutorials/customization/basics
"""

import tempfile
import time

import numpy as np
import tensorflow as tf


def time_matmul(x):
    start = time.time()
    for _ in range(10):
        tf.matmul(x, x)

    result = time.time() - start
    print(f'10 loops: {1000 * result:.2f}ms')


def main():
    # Tensors
    print(f'1 + 2 = {tf.add(1, 2)}')
    print(f'[1, 2] + [3, 4] = {tf.add([1, 2], [3, 4])}')
    print(f'5² = {tf.square(5)}')
    print(f'sum([1, 2, 3]) = {tf.reduce_sum([1, 2, 3])}')

    # Operator overloading is also supported
    print(f'2² + 3² = {tf.square(2) + tf.square(3)}')

    x = tf.matmul([[1]], [[2, 3]])
    print(f'Matrix multiplication: {x}')
    print(f'Matrix shape: {x.shape}')
    print(f'Matrix type: {x.dtype}')

    # NumPy Compatibility
    ndarray = np.ones([3, 3])
    print('TensorFlow operations convert numpy arrays to Tensors automatically')
    tensor = tf.multiply(ndarray, 42)
    print(tensor)

    print('And NumPy operations convert TEnsors to numpy arrays automatically')
    print(np.add(tensor, 1))

    print('The .numpy() method explicitly converts a Tensor to a numpy array')
    print(tensor.numpy())

    # GPU acceleration
    x = tf.random.uniform([3, 3])
    print(f'Is there a GPU available: {tf.config.list_physical_devices("GPU")}')

    print(f'Is there a Tensor on GPU #0: {x.device.endswith("GPU:0")}')

    # Device names
    # Explicit Device Placement

    # Force execution on CPU
    print('On CPU:')
    with tf.device('CPU:0'):
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith('CPU:0')
        time_matmul(x)

    # Foce execution on GPU #0 if available
    if tf.config.list_physical_devices('GPU'):
        print('On GPU:')
        with tf.device('GPU:0'):    # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd, etc.
            x = tf.random.uniform([1000, 1000])
            assert x.device.endswith('GPU:0')
            time_matmul(x)

    # Datasets
    ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

    # Create a CSV file
    _, filename = tempfile.mkstemp()

    with open(filename, 'w') as f:
        f.write('Line1\nLine2\nLine3')

    ds_file = tf.data.TextLineDataset(filename)

    # Apply transformations
    ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
    ds_file = ds_file.batch(2)

    # Iterate
    print('Elements of de_tensors:')
    [print(x) for x in ds_tensors]
    print()
    print(f'Elements in ds_file:')
    [print(x) for x in ds_file]


if __name__ == "__main__":
    main()
