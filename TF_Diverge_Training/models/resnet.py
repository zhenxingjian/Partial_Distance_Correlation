'''
ResNet18/34/50/101/152 in TensorFlow2.

Reference:
[1] He, Kaiming, et al. 
    "Deep residual learning for image recognition." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
'''
import tensorflow as tf
from tensorflow.keras import layers

class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x, return_feats=False):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BuildResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, input_shape=(32, 32, 3), return_feats=False):
        super(BuildResNet, self).__init__()
        self.in_channels = 64
        self.return_feats = return_feats
        if input_shape != (None, 32, 32, 3):  
            self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        else:
            #cifar10
            self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
            
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
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
        features = out
        out = self.fc(out)
        
        if self.return_feats:
            return out, features
        else:
            return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)


def ResNet(model_type, num_classes, input_shape=(None, 32, 32, 3), return_feats=False):
    if model_type == 'resnet18':
        model =  BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_shape, return_feats)
    elif model_type == 'resnet34':
        model = BuildResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_shape, return_feats)
    elif model_type == 'resnet50':
        model =  BuildResNet(BottleNeck, [3, 4, 6, 3], num_classes, input_shape, return_feats)
    elif model_type == 'resnet101':
        model =  BuildResNet(BottleNeck, [3, 4, 23, 3], num_classes, input_shape, return_feats)
    elif model_type == 'resnet152':
        model = BuildResNet(BottleNeck, [3, 8, 36, 3], num_classes, input_shape, return_feats)
    else:
        raise NotImplementedError
    model.build(input_shape)

    return model

def resnet18(num_classes, input_shape=(None, 32, 32, 3), return_feats=False):
    return ResNet('resnet18', num_classes, input_shape, return_feats)
def resnet34(num_classes, input_shape=(None, 32, 32, 3), return_feats=False):
    return ResNet('resnet34', num_classes, input_shape, return_feats)
def resnet50(num_classes, input_shape=(None, 32, 32, 3), return_feats=False):
    return ResNet('resnet50', num_classes, input_shape, return_feats)
def resnet101(num_classes, input_shape=(None, 32, 32, 3), return_feats=False):
    return ResNet('resnet101', num_classes, input_shape, return_feats)
def resnet152(num_classes, input_shape=(None, 32, 32, 3), return_feats=False):
    return ResNet('resnet152', num_classes, input_shape, return_feats)
