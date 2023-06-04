'''
AlexNet in TensorFlow2.

Reference:
[1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. 
    "Imagenet classification with deep convolutional neural networks." 
    Advances in neural information processing systems 25 (2012): 1097-1105.
'''
import tensorflow as tf
from tensorflow.keras import layers

class AlexNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = layers.Conv2D(96, kernel_size=11, strides=4, padding='same', activation='relu')
        self.max_pool2d1 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv2 = layers.Conv2D(256, kernel_size=5, padding='same', activation='relu')
        self.max_pool2d2 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv3 = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')
        self.conv5 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.max_pool2d3 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        out = self.max_pool2d1(self.conv1(x))
        out = self.max_pool2d2(self.conv2(out))
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.max_pool2d3(self.conv5(out))
        out = self.flatten(out)
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.fc3(out)
        return out