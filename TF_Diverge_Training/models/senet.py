'''
SEResNet18/34/50/101/152 in TensorFlow2.
SEPreActResNet18/34/50/101/152 in TensorFlow2.

Reference:
[1] Hu, Jie, Li Shen, and Gang Sun. 
    "Squeeze-and-excitation networks." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
'''
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import sys

class SELayer(tf.keras.Model):
    def __init__(self, out_channels, reduction=16):
        super(SELayer, self).__init__()
        self.adapt_pool2d = tfa.layers.AdaptiveAveragePooling2D(output_size=1)
        self.squeeze = tf.keras.Sequential([
            layers.Conv2D(out_channels//reduction, kernel_size=1, use_bias=False, activation='relu'),
            layers.Conv2D(out_channels, kernel_size=1, use_bias=False, activation='sigmoid')
        ])
        
    def call(self, x):
        # Squeeze
        out = self.adapt_pool2d(x)
        out = self.squeeze(out)
        # Excitation
        out = x * tf.broadcast_to(out, x.shape)
        return out

class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.se = SELayer(self.expansion*out_channels, reduction)
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1, reduction=16):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.se = SELayer(self.expansion*out_channels, reduction)
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class PreActBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1, reduction=16):
        super(PreActBlock, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False)
        self.se = SELayer(self.expansion*out_channels, reduction)
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False)
            ])
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(tf.keras.activations.relu(self.bn2(out)))
        out = self.se(out)
        out = layers.add([shortcut, out])
        return out

class PreActBottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1, reduction=16):
        super(PreActBottleNeck, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.se = SELayer(self.expansion*out_channels, reduction)
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False)
            ])
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(tf.keras.activations.relu(self.bn2(out)))
        out = self.conv3(tf.keras.activations.relu(self.bn3(out)))
        out = self.se(out)
        out = layers.add([shortcut, out])
        return out

class BuildSEResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, reduction=16):
        super(BuildSEResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2, reduction=reduction)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides, reduction):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s, reduction)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)

def SEResNet(model_type, num_classes):
    if model_type == 'seresnet18':
        return BuildSEResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif model_type == 'seresnet34':
        return BuildSEResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif model_type == 'seresnet50':
        return BuildSEResNet(BottleNeck, [3, 4, 6, 3], num_classes)
    elif model_type == 'seresnet101':
        return BuildSEResNet(BottleNeck, [3, 4, 23, 3], num_classes)
    elif model_type == 'seresnet152':
        return BuildSEResNet(BottleNeck, [3, 8, 36, 3], num_classes)
    else:
        sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))

def SEPreActResNet(model_type, num_classes):
    if model_type == 'sepreactresnet18':
        return BuildSEResNet(PreActBlock, [2, 2, 2, 2], num_classes)
    elif model_type == 'sepreactresnet34':
        return BuildSEResNet(PreActBlock, [3, 4, 6, 3], num_classes)
    elif model_type == 'sepreactresnet50':
        return BuildSEResNet(PreActBottleNeck, [3, 4, 6, 3], num_classes)
    elif model_type == 'sepreactresnet101':
        return BuildSEResNet(PreActBottleNeck, [3, 4, 23, 3], num_classes)
    elif model_type == 'sepreactresnet152':
        return BuildSEResNet(PreActBottleNeck, [3, 8, 36, 3], num_classes)
    else:
        sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))
