import os
import sys
import time
import math
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.layers as layers
# import torch.nn as nn
# import torch.nn.init as init


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
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

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
    A MultiStepLR learning rate schedule equivalatent to the Pytorch implementation.
    @param: lr (float): The initial learning rate.
    @param: batch_size (int): The batch size used for training.
    @param: dataset_size (int): The size of the dataset used for training.
    @param: milestones (list): A list of epoch indices at which the learning rate will be multiplied by the gamma value.
    @param: gamma (float): The multiplicative factor to apply to the learning rate at each milestone.
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
        self, initial_learning_rate, decay_steps,steps_per_epoch, alpha=0.0, name=None
    ):
        """Applies cosine decay to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
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
        step = int((step+1)/self.steps_per_epoch)
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


def prepare_cifar10(batch_size, augs: list=[]):
    """
    return preprocessed cifar10 dataset as generators.
    @param augs: list of keras augmentation layers
    """
    print('==> Preparing CIFAR10..')
    train_data = tfds.load("cifar10",split="train",as_supervised=True, data_dir=".")
    val_data = tfds.load("cifar10",split="test",as_supervised=True, data_dir=".")
    augs = keras.Sequential(augs+[
        layers.Rescaling(1./255),
        layers.Normalization(),
    ])
    train_data = train_data.shuffle(1024,reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_data = train_data.map(lambda x,y: (augs(x,training=True),tf.one_hot(y,depth=10)),num_parallel_calls=tf.data.AUTOTUNE)
    
    augs = keras.Sequential([
        layers.Rescaling(1./255),
        layers.Normalization(),
    ])
    val_data = val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = val_data.map(lambda x,y: (augs(x,training=True), tf.one_hot(y,10)),num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_data,val_data
    


def prepare_imagenet(batch_size,augs: list=[]):
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
    train_data, val_data= datasets['train'], datasets['validation']
    assert isinstance(train_data, tf.data.Dataset)
    assert isinstance(val_data, tf.data.Dataset)
    augmentation = keras.Sequential(augs+[
        layers.Resizing(256, 256),
        layers.CenterCrop(224,224),
        layers.Rescaling(1./255),
        layers.Normalization(),
    ])
    train_data = train_data.shuffle(1024,reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_data = train_data.map(lambda x,y: (augmentation(x,training=True), tf.one_hot(y,depth=1000)),num_parallel_calls=tf.data.AUTOTUNE)
    
    val_data = val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = val_data.map(lambda x,y: (augmentation(x,training=True), tf.one_hot(y,depth=1000)),num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_data,val_data
    
    
