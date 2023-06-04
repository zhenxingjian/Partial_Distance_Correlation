import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import os
import argparse
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from absl import app, flags
import numpy as np
from easydict import EasyDict
from utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import *
from training import implemented_nets
from tqdm import tqdm
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.spsa import spsa
from cleverhans.tf2.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

parser = argparse.ArgumentParser(description='Tensorflow CIFAR-10 Training')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.01, type=float, help='balance between accuracy and DC')
parser.add_argument('--eps', default=0.03, type=float, help='eps of adv attack')
parser.add_argument('--num_nets', default=3, type=int, help='number of sub-networks')
parser.add_argument('--network', default='resnet18', type=str, help='name of the network', choices=implemented_nets)
parser.add_argument('--dataset',type=str, choices=["cifar10", "imagenet"],default="cifar10")

args = parser.parse_args()


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = [0 for _ in range(args.num_nets)]  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
if args.dataset=="cifar10":
    train_gen,test_gen = prepare_cifar10(args.batch_size, one_hot=False)
    input_shape = (32, 32, 3)
    num_classes = 10

elif args.dataset=="imagenet":
    train_gen,test_gen,n_train,n_val = prepare_imagenet(args.batch_size, one_hot=False)
    input_shape = (224, 224, 3)
    num_classes = 1000


# Model``
models = [model_hanlder(args.network, input_shape, num_classes, return_feats=True) for _ in range(args.num_nets)]
ckpts = [tf.train.Checkpoint(model=model) for model in models]  
ckpt_paths = [f'./checkpoints/{args.network}/net{i}' for i in range(args.num_nets)]
managers = [tf.train.CheckpointManager(ckpts[i],
            ckpt_paths[i], max_to_keep=1)
            for i in range(args.num_nets)]
for idx, ckpt in enumerate(ckpts):
    ckpt.restore(managers[idx].latest_checkpoint)
    
for model in models:
    model.layers[-1].activation = tf.keras.activations.linear # remove softmax


checkpoint_for_attack = models[1]
checkpoint_for_baseline_eval = models[0]
#attack the third model
checkpoint_for_our_eval = models[2]

class Model_for_attack(keras.Model):
    def __init__(self, net):
        super(Model_for_attack, self).__init__()
        self.net = net

    def call(self, x):
        return self.net(x, training=False)[0] #(outputs, features)


net_for_attack = Model_for_attack(checkpoint_for_attack)
net_for_baseline_eval = Model_for_attack(checkpoint_for_baseline_eval)
net_for_our_eval = Model_for_attack(checkpoint_for_our_eval)

#set up metrics
test_acc_clean = keras.metrics.SparseCategoricalAccuracy()
test_acc_fgm = keras.metrics.SparseCategoricalAccuracy()
test_acc_pgd = keras.metrics.SparseCategoricalAccuracy()

test_acc_clean_div = keras.metrics.SparseCategoricalAccuracy()
test_acc_fgm_div = keras.metrics.SparseCategoricalAccuracy()
test_acc_pgd_div = keras.metrics.SparseCategoricalAccuracy()

### evaluate attack
# Evaluate on clean and adversarial data
print('==> Adversarial attack..')
#net.eval()
report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_spsa=0)

info = "baseline" if args.divergent else "divergent"
# test adv acc
for batch_idx, batch in tqdm(enumerate(test_gen), desc=f"Attacking models", total=len(test_gen)):
        if isinstance(batch, dict):  #grabbed from tfds
            x = batch['image']
            y = batch['label']
        else :
            (x, y) = batch
        x, y = tf.cast(x, tf.float32), tf.cast(y, tf.int64)

        x_fgm = fast_gradient_method(net_for_attack, x, args.eps, np.inf)
        x_pgd = projected_gradient_descent(net_for_attack, x, args.eps, 0.01, 40, np.inf)
        # x_spsa = spsa(net, x, args.eps, 40, sanity_checks=False)
        y_pred = net_for_baseline_eval(x)  # model prediction on clean examples
        y_pred_fgm = net_for_baseline_eval(x_fgm)  # model prediction on FGM adversarial examples
        y_pred_pgd = net_for_baseline_eval(x_pgd)  # model prediction on PGD adversarial examples

        test_acc_clean(y, y_pred)
        test_acc_fgm(y, y_pred_fgm)
        test_acc_pgd(y, y_pred_pgd)


        #models trained with DC
        y_pred_div = net_for_our_eval(x) # model prediction on clean examples
        y_pred_fgm_div = net_for_our_eval(x_fgm)  # model prediction on FGM adversarial examples
        y_pred_pgd_div = net_for_our_eval(x_pgd)  # model prediction on PGD adversarial examples
        # _, y_pred_spsa = net(x_spsa).max(1)
        test_acc_clean_div(y, y_pred_div)
        test_acc_fgm_div(y, y_pred_fgm_div)
        test_acc_pgd_div(y, y_pred_pgd_div)
        # report.correct_spsa += y_pred_spsa.eq(y).sum().item()

print(f'Using {info} models')
print(
        "baseline test acc on clean examples (%): {:.3f}".format(
            test_acc_clean.result() * 100.0
        )
)
print(
"baseline test acc on FGM adversarial examples (%): {:.3f}".format(
    test_acc_fgm.result() * 100.0
)
)
print(
"baseline test acc on PGD adversarial examples (%): {:.3f}".format(
    test_acc_pgd.result() * 100.0
)
)

print(
        "divergent test acc on clean examples (%): {:.3f}".format(
            test_acc_clean_div.result() * 100.0
        )
)
print(
"divergent test acc on FGM adversarial examples (%): {:.3f}".format(
    test_acc_fgm_div.result() * 100.0
)
)
print(
"divergent test acc on PGD adversarial examples (%): {:.3f}".format(
    test_acc_pgd_div.result() * 100.0
)
)
exit()
# print(
# "test acc on SPSA adversarial examples (%): {:.3f}".format(
#     report.correct_spsa / report.nb_test * 100.0
# )
# )
#'''
