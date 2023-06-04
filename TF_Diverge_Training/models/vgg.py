'''
VGG11/13/16/19 in TensorFlow2.

Reference:
[1] Simonyan, Karen, and Andrew Zisserman. 
    "Very deep convolutional networks for large-scale image recognition." 
    arXiv preprint arXiv:1409.1556 (2014).
'''
import tensorflow as tf
from tensorflow.keras import layers

config = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(tf.keras.Model):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.conv = self._make_layers(config[vgg_name])
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layers(self, config):
        layer = []
        for l in config:
            if l == 'M':
                layer += [layers.MaxPool2D(pool_size=2, strides=2)]
            else:
                layer += [layers.Conv2D(l, kernel_size=3, padding='same'),
                          layers.BatchNormalization(),
                          layers.ReLU()]
        layer += [layers.AveragePooling2D(pool_size=1, strides=1)]
        return tf.keras.Sequential(layer)
