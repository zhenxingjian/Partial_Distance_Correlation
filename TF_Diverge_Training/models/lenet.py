'''
LeNet in TensorFlow2.

Reference:
[1] LeCun, Yann, et al. 
    "Gradient-based learning applied to document recognition." 
    Proceedings of the IEEE 86.11 (1998): 2278-2324.
'''
import tensorflow as tf
from tensorflow.keras import layers

class LeNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = layers.Conv2D(6, kernel_size=5, activation='sigmoid')
        self.conv2 = layers.Conv2D(16, kernel_size=5, activation='sigmoid')
        self.max_pool2d = layers.MaxPool2D(pool_size=2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='sigmoid')
        self.fc2 = layers.Dense(84, activation='sigmoid')
        self.fc3 = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x):
        out = self.conv1(x)
        out = self.max_pool2d(out)
        out = self.conv2(out)
        out = self.max_pool2d(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out