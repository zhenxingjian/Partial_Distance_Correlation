"""
This is the main divergr training module for CIFAR10 and ImageNet.
By default it will train ResNet18 on CIFAR10 to produce results in the paper.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import albumentations as A
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
import tensorflow.keras as keras
# from torchvision import datasets
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import argparse
import json
from DC_criterion import Loss_DC,run_nets
from utils import *
from Resnet import *
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.05, type=float, help='balance between accuracy and DC')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_nets', default=3, type=int, help='number of sub-networks')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# parser.add_argument('--workers', default=2, type=int, help='number of workers for dataloader')
parser.add_argument('--network', default='resnet152', type=str, help='name of the network')
parser.add_argument('--epochs',type=int,help="training epochs. 200 according to the paper.",default=200)
parser.add_argument('--dataset',type=str,help="cifar10 or imagenet",default="cifar10")
args = parser.parse_args()

# def preprocess(x,y):
#     """
#     Normalize x and convert y to one-hot.
#     @param x (np.ndarray): Image data array to be normalized channel-wise, shape should be (batch_size,H,W,C)
#     @param y (np.ndarray): Labels. Shape should be (batch_size,)
#     """
#     assert len(x.shape)>=4, "Can only normalize batched 3-channel images!"
#     assert len(y.shape)==2, "Incorret label shape!"
#     assert len(x)==len(y), "Number of labels doesn't match number of images"
#     x = x.astype("float32")
#     x = x.astype("float32")
#     x /= 255 
#     return (x-x.mean(axis=(0,1,2)))/x.std(axis=(0,1,2)), to_categorical(y) 

best_acc = [0 for _ in range(args.num_nets)]  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



# Get dataset
if args.dataset=="cifar10":
    train_gen,test_gen = prepare_cifar10(args.batch_size)
    scheduler = tf.keras.optimizers.schedules.CosineDecay(
    args.lr, decay_steps=120, alpha=0.0, name=None
)
elif args.dataset=="imagenet":
    train_gen,test_gen,n_train,n_val = prepare_imagenet(args.batch_size)
    #My customized MultiStepLR. Keras doesn't implement this.
    scheduler = MultiStepLR(
                            lr=args.lr,batch_size=args.batch_size,
                            dataset_size=n_train,milestones=[10,20,30],
                            gamma=0.1
                            )
# Model
print('==> Building model..')
#net = ResNet18(num_nets = args.num_nets)
if args.network == 'resnet152':
    model_name = ResNet152
elif args.network == 'resnet34':
    model_name = ResNet34
elif args.network == 'resnet18':
    model_name = ResNet18
elif args.network == 'resnet101':
    model_name= ResNet101
if args.dataset=='cifar10':
    net = [model_name(input_shape=(32,32,3),classes=10) for _ in range(args.num_nets)]
else:
    net = [model_name(input_shape=(224,224,3),classes=10) for _ in range(args.num_nets)]
optimizers = [keras.optimizers.SGD(learning_rate=scheduler,momentum=0.9,decay=5e-4) for _ in range(args.num_nets)]
model_paths = []

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists('./checkpoint/ckpt_'+str(args.num_nets-1))
    for idx in range(args.num_nets):
        model_paths.append('./checkpoint/ckpt_'+args.network+"_"+str(idx)+"_"+args.dataset)
        # checkpoint = torch.load(model_path)
        log = json.load(model_paths[idx]+'_log.json')
        # net[idx].load_state_dict(checkpoint['net'])
        net[idx] = keras.models.load_model(model_paths[idx])
        best_acc[idx] = log['acc']
        start_epoch = max(start_epoch, log['epoch'])
        #set up otimizer
        optim_weights = np.load(model_paths[idx]+"_optimizer.npy")
        optimizers[idx].set_weights(optim_weights)



criterion = Loss_DC(alpha = args.alpha)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    for idx in range(args.num_nets):
        train_loss = 0
        correct = 0
        total = 0

        DC_results_total = np.zeros(args.num_nets-1)
        #enumerate(train_gen) will iterate endlessly
        for batch_idx in range(len(train_gen)):
            batch = next(train_gen)
            if isinstance(batch,dict):  #grabbed from tfds
                inputs = batch['image']
                targets = batch['label']
            else :
                (inputs, targets) = batch#keras ImageDataGenerator

            with tf.GradientTape() as tape:
                outputs, loss, DC_results = run_nets(net, idx, inputs, targets, criterion, args)
                grads = tape.gradient(loss,net[idx].trainable_weights)
                optimizers[idx].apply_gradients(zip(grads,net[idx].trainable_weights))
                train_loss += loss.numpy()

            predicted = tf.math.argmax(outputs,axis=1)
            targets = tf.math.argmax(targets,axis=1)
            total += len(targets)
            # if total==0:
            #     print(f"total=0!!!batch size{batch_idx}")
            #     print(targets)
            correct += tf.math.count_nonzero(tf.math.equal(predicted,(targets)))
            DC_results_total += DC_results
            
            if batch_idx>391:
                print('bug')
            progress_bar(batch_idx, len(train_gen), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f'
                        % (train_loss/(batch_idx+1), 100.*correct.numpy()/total, correct, total, 
                            DC_results_total[0]/(batch_idx+1) , DC_results_total[1]/(batch_idx+1) ))
            batch_idx += 1
            train_gen.on_epoch_end()



def test(epoch):
    global best_acc
    current_acc = []
    for idx in range(args.num_nets):
        test_loss = 0
        correct = 0
        total = 0
        DC_results_total = np.zeros(args.num_nets-1)

        for batch_idx in range(len(test_gen)):
            batch = next(test_gen)
            if isinstance(batch,dict):   #grab data from tfds
                inputs = batch['image']
                targets = batch['label']
            else :
                (inputs, targets) = batch#grab data from keras ImageDataGenerator
                outputs, loss, DC_results = run_nets(net, idx, inputs, targets, criterion, args,train=False)
                loss = criterion.CE(outputs, targets)

                test_loss += loss.numpy()
                predicted = tf.math.argmax(outputs,axis=1)
                targets = tf.math.argmax(targets,axis=1)

                total += len(targets)
                correct += tf.math.count_nonzero(tf.math.equal(predicted,(targets)))

                DC_results_total += DC_results

                progress_bar(batch_idx, len(test_gen), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f'
                            % (test_loss/(batch_idx+1), 100.*correct.numpy()/total, correct, total, 
                            DC_results_total[0]/(batch_idx+1) , DC_results_total[1]/(batch_idx+1) ))
                batch_idx += 1
                test_gen.on_epoch_end()

        # Save checkpoint.
        acc = 100.*correct/total
        current_acc.append(acc)

    if sum(current_acc) > sum(best_acc):
        print('Saving..')
        for idx in range(args.num_nets):
            state = {
                'net': net[idx].state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            net[idx].save(model_paths[idx])
            json.dump(state,open(model_paths[idx]+"_log.json","wb"))
            np.save(model_paths[idx]+"_optimizer.npy",optimizers[idx].get_weights())
            best_acc[idx] = current_acc[idx]


for epoch in range(start_epoch, args.epochs):
    train(epoch)
    test(epoch)

