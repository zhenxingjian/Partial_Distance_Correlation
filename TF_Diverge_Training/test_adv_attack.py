import torch
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
from TF_Diverge_Training.models import resnet34, resnet152,resnet18,resnet101,ResNet50


from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.spsa import spsa
from cleverhans.tf2.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

parser = argparse.ArgumentParser(description='Tensorflow CIFAR-10 Training')
parser.add_argument('--divergent', default=1, type=int, help='whether use digergent model or baseline model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.01, type=float, help='balance between accuracy and DC')
parser.add_argument('--eps', default=0.3, type=float, help='eps of adv attack')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_nets', default=3, type=int, help='number of sub-networks')
# parser.add_argument('--workers', default=2, type=int, help='number of wokers')
parser.add_argument('--network', default='resnet152', type=str, help='name of the network')
parser.add_argument('--dataset',type=str,help="cifar10 or imagenet",default="cifar10")

args = parser.parse_args()


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = [0 for _ in range(args.num_nets)]  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

if args.dataset=="cifar10":
    train_gen,test_gen = prepare_cifar10(args.batch_size)
    scheduler = tf.keras.optimizers.schedules.CosineDecay(
    args.lr, decay_steps=120, alpha=0.0, name=None
)
elif args.dataset=="imagenet":
    train_gen,test_gen,n_train,n_val = prepare_imagenet(args.batch_size)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model``
checkpoint_for_attack = keras.model.load_model('./checkpoint/ckpt_'+args.network+"_"+str(1)+'_'+args.dataset)
checkpoint_for_baseline_eval = keras.model.load_model('./checkpoint/ckpt_'+args.network+"_"+str(0)+'_'+args.dataset)
checkpoint_for_our_eval = keras.model.load_model('./checkpoint/ckpt_'+args.network+"_"+str(2)+'_'+args.dataset)


##### test infenrence different with attack model
class Transfer(keras.Model):
    def __init__(self, net):
        super(Transfer, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net.predict(x)[0]


net_for_attack = Transfer(checkpoint_for_attack)
net_for_baseline_eval = Transfer(checkpoint_for_baseline_eval)
net_for_our_eval = Transfer(checkpoint_for_our_eval)



### evaluate attack
# Evaluate on clean and adversarial data
print('==> Adversarial attack..')
#net.eval()
report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_spsa=0)



#'''
# test adv acc
for batch_idx, batch in enumerate(test_gen):
        if isinstance(batch,dict):  #grabbed from tfds
            x = batch['image']
            y = batch['label']
        else :
            (x, y) = batch#keras ImageDataGenerator
        x_fgm = fast_gradient_method(net_for_attack, x, args.eps, np.inf)
        x_pgd = projected_gradient_descent(net_for_attack, x, args.eps, 0.01, 40, np.inf)
        #x_spsa = spsa(net, x, args.eps, 40, sanity_checks=False)
        if args.divergent:
            y_pred = tf.math.argmax(net_for_our_eval(x),axis=1)  # model prediction on clean examples
            y_pred_fgm = tf.math.argmax(net_for_our_eval(x_fgm),axis=1)  # model prediction on FGM adversarial examples
            y_pred_pgd = tf.math.argmax(net_for_our_eval(x_pgd),axis=1)  # model prediction on PGD adversarial examples
        else:
            y_pred = tf.math.argmax(net_for_baseline_eval(x),axis=1)  # model prediction on clean examples
            y_pred_fgm = tf.math.argmax(net_for_baseline_eval(x_fgm),axis=1)  # model prediction on FGM adversarial examples
            y_pred_pgd = tf.math.argmax(net_for_baseline_eval(x_pgd),axis=1)  # model prediction on PGD adversarial examples
        #_, y_pred_spsa = net(x_spsa).max(1)
        report.nb_test += len(y)
        report.correct += tf.math.count_nonzero(tf.math.equal(y_pred,y)).numpy()
        report.correct_fgm += tf.math.count_nonzero(tf.math.equal(y_pred_fgm,y)).numpy()
        report.correct_pgd += tf.math.count_nonzero(tf.math.equal(y_pred_pgd,y)).numpy()
        #report.correct_spsa += y_pred_spsa.eq(y).sum().item()
        if batch_idx % 10 == 0:
            print('Processing batch_idx:', batch_idx)
print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
)
print(
"test acc on FGM adversarial examples (%): {:.3f}".format(
    report.correct_fgm / report.nb_test * 100.0
)
)
print(
"test acc on PGD adversarial examples (%): {:.3f}".format(
    report.correct_pgd / report.nb_test * 100.0
)
)
print(
"test acc on SPSA adversarial examples (%): {:.3f}".format(
    report.correct_spsa / report.nb_test * 100.0
)
)
#'''
