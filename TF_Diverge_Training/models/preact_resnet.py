# +
'''
PreActResNet18/34/50/101/152 in TensorFlow2.

Reference:
[1] He, Kaiming, et al. 
    "Identity mappings in deep residual networks." 
    European conference on computer vision. Springer, Cham, 2016.
'''
import tensorflow as tf
from tensorflow.keras import layers
import sys

class PreActBlock(tf.keras.Model):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(PreActBlock, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False)
            ])
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(tf.keras.activations.relu(self.bn2(out)))
        out = layers.add([shortcut, out])
        return out


# -

class PreActBottleNeck(tf.keras.Model):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(PreActBottleNeck, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        
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
        out = layers.add([shortcut, out])
        return out

class BuildPreActResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes):
        super(BuildPreActResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)

def PreActResNet(model_type, num_classes):
    if model_type == 'preactresnet18':
        return BuildPreActResNet(PreActBlock, [2, 2, 2, 2], num_classes)
    elif model_type == 'preactresnet34':
        return BuildPreActResNet(PreActBlock, [3, 4, 6, 3], num_classes)
    elif model_type == 'preactresnet50':
        return BuildPreActResNet(PreActBottleNeck, [3, 4, 6, 3], num_classes)
    elif model_type == 'preactresnet101':
        return BuildPreActResNet(PreActBottleNeck, [3, 4, 23, 3], num_classes)
    elif model_type == 'preactresnet152':
        return BuildPreActResNet(PreActBottleNeck, [3, 8, 36, 3], num_classes)
    else:
        sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))
