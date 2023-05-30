"""
Some helper functions for TensorFlow2.0, including:
    - get_dataset(): download dataset from TensorFlow.
    - get_mean_and_std(): calculate the mean and std value of dataset.
    - normalize(): normalize dataset with the mean the std.
    - dataset_generator(): return `Dataset`.
    - progress_bar(): progress bar mimic xlua.progress.
"""
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np

padding = 4
image_size = 32
target_size = image_size + padding*2

def get_dataset():
    """Download, parse and process a dataset to unit scale and one-hot labels."""
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images/255.0, test_images/255.0
    
    # One-hot labels
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)
    return train_images, train_labels, test_images, test_labels

def get_mean_and_std(images):
    """Compute the mean and std value of dataset."""
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std

def normalize(images, mean, std):
    """Normalize data with mean and std."""
    return (images - mean) / std

def dataset_generator(images, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return ds


def _one_hot(train_labels, num_classes, dtype=np.float32):
    """Create a one-hot encoding of labels of size num_classes."""
    return np.array(train_labels == np.arange(num_classes), dtype)

def _augment_fn(images, labels):
    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels