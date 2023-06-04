'''
MobileNet in TensorFlow2.

Reference:
[1] Sandler, Mark, et al. 
    "Mobilenetv2: Inverted residuals and linear bottlenecks." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
'''
import tensorflow as tf
from tensorflow.keras import layers

class Block(tf.keras.Model):
    '''Expand + depthwise & pointwise convolution'''
    def __init__(self, in_channels, out_channels, expansion, strides):
        super(Block, self).__init__()
        self.strides = strides
        channels = expansion * in_channels
        
        self.conv1 = layers.Conv2D(channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(channels, kernel_size=3, strides=strides, padding='same',
                                   groups=channels, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides == 1 and in_channels != out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=1, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
        
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out]) if self.strides==1 else out
        return out

class MobileNetV2(tf.keras.Model):
    # (expansion, out_channels, num_blocks, strides)
    config = [(1, 16, 1, 1),
              (6, 24, 2, 1),  # NOTE: change strides 2 -> 1 for CIFAR10
              (6, 32, 3, 2),
              (6, 64, 4, 2),
              (6, 96, 3, 1),
              (6, 160, 3, 2),
              (6, 320, 1, 1)]
    
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 strides 2 -> 1 for CIFAR10
        self.conv1 = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer = self._make_layers(in_channels=32)
        self.conv2 = layers.Conv2D(1280, kernel_size=1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer(out)
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
        
    def _make_layers(self, in_channels):
        layer = []
        for expansion, out_channels, num_blocks, strides in self.config:
            stride = [strides] + [1]*(num_blocks-1)
            for s in stride:
                layer += [Block(in_channels, out_channels, expansion, s)]
                in_channels = out_channels
        return tf.keras.Sequential(layer)
