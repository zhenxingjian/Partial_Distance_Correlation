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
from easydict import EasyDict
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.models as models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Resnet import ResNet34, ResNet152,ResNet18,ResNet101,ResNet50


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
args = parser.parse_args()


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = [0 for _ in range(args.num_nets)]  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

(x_train,y_train),(x_test,y_test) = cifar10.load_data()


train_gen =  ImageDataGenerator(
    horizontal_flip = True,
    featurewise_center = True,
    featurewise_std_normalization = True,
    rescale = True
)
test_gen =  ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    rescale = True
)
train_gen.fit(x_train)
test_gen.fit(x_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.network == 'resnet152':
    model_name = ResNet152
elif args.network == 'resnet34':
    model_name = ResNet34
elif args.network == 'resnet18':
    model_name = ResNet18
elif args.network == 'resnet101':
    model_name= ResNet101

net = [model_name(input_shape=(32,32,3),classes=10) for _ in range(args.num_nets)]

# net = [ subnet.to(device) for subnet in net ]
# if device == 'cuda':
#     net = [torch.nn.DataParallel(subnet) for subnet in net]
#     cudnn.benchmark = True

# define the ensemble network
class Ensemble(nn.Module):
    def __init__(self, net_list):
        super(Ensemble, self).__init__()
        self.net_list = net_list

    def forward(self, x):
        random_idx = random.randint(0,2)
        return self.net_list[0](x)[0] #torch.stack([self.net_list[idx](x)[0] for idx in range(len(self.net_list))], 0).mean(0)

'''
# breakpoint()
if args.divergent: #args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists('./checkpoint/ckpt_'+str(args.num_nets-1)+'.pth')
    for idx in range(args.num_nets):
        checkpoint = torch.load('./checkpoint/ckpt_'+str(idx)+'.pth')
        net[idx].load_state_dict(checkpoint['net'])
else:
    # Load checkpoint.
    print('==> Resuming from baseline checkpoint..')
    assert os.path.isdir('../%s_baseline_on_imagenet/checkpoint'%network_name), 'Error: no checkpoint directory found!'
    assert os.path.exists('../%s_baseline_on_imagenet/checkpoint/ckpt_'%network_name+str(args.num_nets-1)+'.pth')
    for idx in range(args.num_nets):
        checkpoint = torch.load('../%s_baseline_on_imagenet/checkpoint/ckpt_'%network_name+str(idx)+'.pth')
        net[idx].load_state_dict(checkpoint['net'])
'''
### my load net
checkpoint_for_attack = keras.model.load_model('./checkpoint/%s_ckpt_'%args.network+str(1))
checkpoint_for_baseline_eval = keras.model.load_model('./checkpoint/%s_ckpt_'%args.network+str(0))
checkpoint_for_our_eval = keras.model.load_model('./checkpoint/%s_ckpt_'%args.network+str(2))


##### test infenrence different with attack model
class Transfer(nn.Module):
    def __init__(self, net):
        super(Transfer, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[0]

for i in range(len(net)):
    net[i].eval()
net_for_attack = Transfer(net[0])
net_for_baseline_eval = Transfer(net[1])
net_for_our_eval = Transfer(net[2])

#ensemble them
#net = Ensemble(net)
#net.to(device)


### evaluate attack
# Evaluate on clean and adversarial data
print('==> Adversarial attack..')
#net.eval()
report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_spsa=0)



#'''
# test adv acc
for batch_idx, (x, y) in enumerate(testloader):
        x_fgm = fast_gradient_method(net_for_attack, x, args.eps, np.inf)
        x_pgd = projected_gradient_descent(net_for_attack, x, args.eps, 0.01, 40, np.inf)
        #x_spsa = spsa(net, x, args.eps, 40, sanity_checks=False)
        if args.divergent:
            _, y_pred = net_for_our_eval(x).max(1)  # model prediction on clean examples
            _, y_pred_fgm = net_for_our_eval(x_fgm).max(
                1
            )  # model prediction on FGM adversarial examples
            _, y_pred_pgd = net_for_our_eval(x_pgd).max(
                1
            )  # model prediction on PGD adversarial examples
        else:
            _, y_pred = net_for_baseline_eval(x).max(1)  # model prediction on clean examples
            _, y_pred_fgm = net_for_baseline_eval(x_fgm).max(
                1
            )  # model prediction on FGM adversarial examples
            _, y_pred_pgd = net_for_baseline_eval(x_pgd).max(
                1
            )  # model prediction on PGD adversarial examples
        #_, y_pred_spsa = net(x_spsa).max(1)
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
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
