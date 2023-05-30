import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from models import resnet34
from utils import *
# from tensorflow.keras.optimizers.schedules import CosineDecay

batch_size = 128


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return tf.convert_to_tensor(optimizer._learning_rate(optimizer.iterations)) # I use ._decayed_lr method instead of .lr
    return lr

train_gen, test_gen = prepare_cifar10(batch_size, augs= [
                                                        #layers.Reshape(input_shape,input_shape=input_shape), 
                                                        layers.ZeroPadding2D(4,"channels_last"),
                                                        layers.RandomCrop(32,32), 
                                                        layers.RandomFlip("horizontal")
                                                        ]) 

scheduler = CosineDecay(
                            0.1,
                            steps_per_epoch=len(train_gen),
                            decay_steps=200,
                            alpha=0.0,
)

model = resnet34(classes=10, input_shape=(32, 32, 3), get_features=False)
optim = keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9, nesterov=True)
# Compile the model
model.compile(optimizer=optim,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy', get_lr_metric(optim)],
              )

# Train the model
model.fit(train_gen, validation_data=test_gen, epochs=200)
