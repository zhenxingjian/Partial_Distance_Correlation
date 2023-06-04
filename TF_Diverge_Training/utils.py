import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import multipledispatch
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import os
import sys
import time
import math
from models import *
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.layers as layers


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return tf.convert_to_tensor(optimizer._learning_rate(optimizer.iterations)) # I use ._decayed_lr method instead of .lr
    return lr


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 45.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None, stdout=sys.stdout):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    stdout.write(' [')
    for i in range(cur_len):
        stdout.write('=')
    stdout.write('>')
    for i in range(rest_len):
        stdout.write('.')
    stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []

    if msg:
        L.append(' | ' + msg)
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    
    msg = ''.join(L)
    stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        stdout.write('\b')
    stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        stdout.write('\r')
    else:
        stdout.write('\n')
    stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class MultiStepLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A MultiStep learning rate scheduler. 
    Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones. 
    Args:
        lr (float): The initial learning rate.
        batch_size (int): The batch size used for training.
        dataset_size (int): The size of the dataset used for training.
        milestones (list): A list of epoch indices at which the learning rate will be multiplied by the gamma value.
        gamma (float): The multiplicative factor to apply to the learning rate at each milestone.
    """
    def __init__(self, lr,steps_per_epoch,milestones:list,gamma:int):
        self.lr = lr
        self.epoch = 0
        self.steps_per_epoch = self.steps_per_epoch
        self.milestone = milestones
        self.gamma = gamma

    def __call__(self, step):
        self.epoch = int((step+1)/self.steps_per_epoch)
        lr = self.lr
        for milestone in self.milestones:
            if self.epoch >= milestone:
                lr *= self.gamma
        return lr


#modified implementation to decay by epochs
class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule.
    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.
    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function
    to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
      decayed = (1 - alpha) * cosine_decay + alpha
      return initial_learning_rate * decayed
    ```
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self, initial_learning_rate, decay_steps, steps_per_epoch, alpha=0.0, name=None
    ):
        """Applies cosine decay to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps (epochs as per the original paper: https://arxiv.org/abs/1608.03983) to decay over.
          steps_per_epoch: specifies the number of steps (batches) per epoch. Neccessary for epoch-wise decay.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of initial_learning_rate.
          name: String. Optional name of the operation.  Defaults to
            'CosineDecay'.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.steps_per_epoch = steps_per_epoch

    def __call__(self, step):
        step = (step+1) / self.steps_per_epoch
        with tf.name_scope(self.name or "CosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)

            global_step_recomp = tf.cast(step, dtype)
            global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
            completed_fraction = global_step_recomp / decay_steps
            cosine_decayed = 0.5 * (
                1.0
                + tf.cos(tf.constant(math.pi, dtype=dtype) * completed_fraction)
            )

            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(initial_learning_rate, decayed)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
        }


def model_hanlder(model_type, input_shape, num_classes, return_feats=False):
    if len(input_shape) == 3:
        input_shape = (None, *input_shape)
    if 'lenet' in model_type:
        return LeNet(num_classes)
    elif 'alexnet' in model_type:
        return AlexNet(num_classes)
    elif 'vgg' in model_type:
        return VGG(model_type, num_classes)
    elif 'resnet' in model_type:
        if 'se' in model_type:
            if 'preact' in model_type:
                return SEPreActResNet(model_type, num_classes)
            else:
                return SEResNet(model_type, num_classes)
        else:
            if 'preact' in model_type:
                return PreActResNet(model_type, num_classes)
            else:
                return ResNet(model_type, num_classes, input_shape, return_feats)
    elif 'densenet' in model_type:
        return DenseNet(model_type, num_classes)
    elif 'mobilenet' in model_type:
        if 'v2' not in model_type:
            return MobileNet(num_classes)
        else:
            return MobileNetV2(num_classes)
    else:
        sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))

        
def get_cifar10():
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


def dataset_generator(images, labels, batch_size, aug=False):

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if aug:
        ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return ds


def dataset_mapper(ds, batch_size, aug=False):
    if aug:
        ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(ds)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def _one_hot(train_labels, num_classes, dtype=np.float32):
    """Create a one-hot encoding of labels of size num_classes."""
    return np.array(train_labels == np.arange(num_classes), dtype)

def _augment_fn(images, labels):
    padding = 4
    image_size = 32
    target_size = image_size + padding*2

    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels

def prepare_cifar10(batch_size, aug=False):

    train_images, train_labels, test_images, test_labels = get_cifar10()
    mean, std = get_mean_and_std(train_images)
    train_images = normalize(train_images, mean, std)
    test_images = normalize(test_images, mean, std)

    train_gen = dataset_generator(train_images, train_labels, batch_size, aug)
    test_gen= tf.data.Dataset.from_tensor_slices((test_images, test_labels)).\
            batch(batch_size * 4).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_gen, test_gen



def prepare_imagenet(batch_size, augs: list=[]):
    """
    return preprocessed ImageNet dataset.
    @param augs: list of keras augmentation layers
    """
    print('==> Preparing ImageNet..')

    ## fetch imagenet dataset directly
    imagenet = tfds.image.Imagenet2012()

    ## describe the dataset with DatasetInfo
    C = imagenet.info.features['label'].num_classes
    n_train = imagenet.info.splits['train'].num_examples
    n_validation = imagenet.info.splits['validation'].num_examples

    assert C == 1000
    assert n_train == 1281167
    assert n_validation == 50000
    imagenet.download_and_prepare()

    # load imagenet data from disk as tf.data.Datasets
    datasets = imagenet.as_dataset()
    train_gen, test_gen= datasets['train'], datasets['validation']
    assert isinstance(train_gen, tf.data.Dataset)
    assert isinstance(test_gen, tf.data.Dataset)
    augmentation = keras.Sequential(augs + [
        layers.Resizing(256, 256),
        layers.CenterCrop(224,224),
        layers.Rescaling(1./255),
    ])
    train_gen = train_gen.shuffle(1024,reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_gen = train_gen.map(lambda x,y: (augmentation(x,training=True), tf.one_hot(y,depth=1000)),num_parallel_calls=tf.data.AUTOTUNE)
    
    test_gen = test_gen.batch(batch_size * 2).prefetch(tf.data.AUTOTUNE)
    test_gen = test_gen.map(lambda x,y: (augmentation(x,training=True), tf.one_hot(y,depth=1000)),num_parallel_calls=tf.data.AUTOTUNE)
    
    #normalize
    norm = layers.Normalization(axis=-1)
    norm.adapt(train_gen)
    train_gen = train_gen.map(lambda x,y: (norm(x),y),num_parallel_calls=tf.data.AUTOTUNE)
    norm.adapt(test_gen)
    test_gen = test_gen.map(lambda x,y: (norm(x),y),num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_gen, test_gen