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

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

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
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

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
    Initializes a MultiStepLR learning rate schedule.
    @param: lr (float): The initial learning rate.
    @param: batch_size (int): The batch size used for training.
    @param: dataset_size (int): The size of the dataset used for training.
    @param: milestones (list): A list of epoch indices at which the learning rate will be multiplied by the gamma value.
    @param: gamma (float): The multiplicative factor to apply to the learning rate at each milestone.
    """
    def __init__(self, lr,batch_size,dataset_size,milestones:list,gamma:int):
        self.lr = lr
        self.epoch = 0
        self.steps_per_epoch = self.dataset_size // self.batch_size
        self.milestone = milestones
        self.gamma = gamma
    def __call__(self, step):
        self.epoch = int((step+1)/self.steps_per_epoch)
        lr = self.lr
        for milestone in self.milestones:
            if self.epoch >= milestone:
                lr *= self.gamma
        return lr

def prepare_cifar10(batch_size):
    print('==> Preparing CIFAR10..')

    (x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()


    train_gen =  keras.preprocessing.image.ImageDataGenerator(
        # horizontal_flip = True,
        featurewise_center = True,
        featurewise_std_normalization = True,
        rescale = True
    )
    test_gen =  keras.preprocessing.image.ImageDataGenerator(
        featurewise_center = True,
        featurewise_std_normalization = True,
        rescale = True
    )
    train_gen.fit(x_train)
    test_gen.fit(x_test)
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    return train_gen.flow(x_train,y_train,batch_size=batch_size),test_gen.flow(x_test,y_test,batch_size=batch_size)


def prepare_imagenet(batch_size):
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
    imagenet.download_and_prepare()   ## need more space in harddrive

    # load imagenet data from disk as tf.data.Datasets
    datasets = imagenet.as_dataset()
    train_data, val_data= datasets['train'], datasets['validation']
    assert isinstance(train_data, tf.data.Dataset)
    assert isinstance(val_data, tf.data.Dataset)
    augmentation = keras.Sequential([
        layers.Resizing(256, 256),#implement augmentation according to the original training scheme
        layers.CenterCrop(224,224),
        layers.Rescaling(1./255),
        layers.Normalization(),
    ])

    train_data = train_data.map(lambda x,y: (augmentation(x,training=True), y))
    train_data = train_data.shuffle(1024,reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_data = val_data.map(lambda x,y: (augmentation(x,training=True), y))
    val_data = val_data.shuffle(1024,reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return tfds.as_numpy(train_data),tfds.as_numpy(val_data),n_train,n_validation

    
    