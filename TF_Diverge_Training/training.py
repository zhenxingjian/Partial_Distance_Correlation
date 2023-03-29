"""
This is the main divergr training module for CIFAR10 and ImageNet.
By default it will train ResNet18 on CIFAR10 to produce results in the paper.
"""
import albumentations as A
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
from DC_criterion import Loss_DC,run_nets,eval_nets
from utils import *
from models import *
import tensorflow.keras.layers as layers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.05, type=float, help='balance between accuracy and DC')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_nets', default=3, type=int, help='number of sub-networks')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# parser.add_argument('--workers', default=2, type=int, help='number of workers for dataloader')
parser.add_argument('--network', default='resnet18', type=str, help='name of the network')
parser.add_argument('--epochs',type=int,help="training epochs. 200 according to the paper.",default=200)
parser.add_argument('--dataset',type=str,help="cifar10 or imagenet",default="cifar10")
parser.add_argument('--log_loss', "-l", type=eval, default=True, help='log training and validation loss in txt files')
args = parser.parse_args()

implemented_nets = ["resnet18", "resnet34", "resnet101", "resnet152"] #add to this list if you implement more models in models.py

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
    #train_gen,test_gen = prepare_cifar10(args.batch_size)
    input_shape = (32,32,3)
    train_gen,test_gen = prepare_cifar10(args.batch_size, augs= [
                                                                #layers.Reshape(input_shape,input_shape=input_shape), 
                                                                layers.ZeroPadding2D(4,"channels_last"),
                                                                layers.RandomCrop(32,32), 
                                                                layers.RandomFlip("horizontal")
                                                                ]) 
    scheduler = CosineDecay(
                            args.lr,
                            steps_per_epoch=len(train_gen),
                            decay_steps=100,
                            alpha=0.0,
                            name=None
)#decay every epoch
elif args.dataset=="imagenet":
    train_gen,test_gen = prepare_imagenet(args.batch_size)
    #My customized MultiStepLR. Keras doesn't implement this.
    scheduler = MultiStepLR(
                            lr=args.lr,
                            steps_per_epoch=len(train_gen),
                            milestones=[10,20,30],
                            gamma=0.1
                            )
# Model
print('==> Building model..')
if args.network in implemented_nets:
    model_name = eval(args.network) #functional programming :) :)
else:
    raise NotImplementedError("Network %s not implemented! Available options are " % args.network, implemented_nets)

if args.dataset=='cifar10':
    nets = [model_name(input_shape=(32,32,3),classes=10) for _ in range(args.num_nets)]
else:
    nets = [model_name(input_shape=(224,224,3),classes=10) for _ in range(args.num_nets)]
try:
    optims = [keras.optimizers.SGD(learning_rate=scheduler,momentum=0.9,decay=5e-4) for _ in range(args.num_nets)]
except:
    optims = [keras.optimizers.SGD(learning_rate=scheduler,momentum=0.9,weight_decay=5e-4) for _ in range(args.num_nets)]
    
model_paths = [f'./checkpoints/{args.network}/ckpt_' + str(idx) + "_" + args.dataset for idx in range(args.num_nets) ]
optim_paths = [os.path.join(model_paths[idx], "optim") for idx in range(args.num_nets)]
checkpoints = [tf.train.Checkpoint(optmizer=optims[idx]) for idx in range(args.num_nets)]

nets[0].summary()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoints'), 'Error: cannot find directory "checkpoints"!'
    
    for idx in range(args.num_nets):
        assert os.path.isdir(model_paths[idx]), f'Error: no checkpoint directory found at {model_paths[idx]} !'
        
        log = json.load(model_paths[idx]+'_log.json')
        nets[idx] = keras.models.load_model(model_paths[idx])
        best_acc[idx] = log['acc']
        start_epoch = max(start_epoch, log['epoch'])
        statuses = [checkpoints[idx].restore(tf.train.latest_checkpoint(model_paths[idx])) for idx in range(args.num_nets)]
        statuses[idx].assert_consumed() # optional sanity check

criterion = Loss_DC(alpha = args.alpha)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    for idx in range(args.num_nets):
        train_loss = 0
        correct = 0
        total = 0

        DC_results_total = np.zeros(args.num_nets-1)
        for batch_idx, (inputs,targets) in enumerate(train_gen):
            outputs, loss, DC_results,grads = run_nets(nets, idx, inputs, targets, criterion)
            optims[idx].apply_gradients(zip(grads,nets[idx].trainable_weights))

            train_loss += loss.numpy()
            predicted = tf.math.argmax(outputs, axis=1)
            targets = tf.math.argmax(targets, axis=1)
            total += len(targets)
            correct += tf.math.count_nonzero(tf.math.equal(predicted, (targets))).numpy()
            DC_results_total += DC_results
            progress_bar(batch_idx, len(train_gen), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f'
                        % (loss, 100.*correct/total, correct, total, 
                            DC_results_total[0]/(batch_idx+1), DC_results_total[1]/(batch_idx+1)))
            #log loss 
            if args.log_loss and batch_idx == len(train_gen)-1:
                os.makedirs(model_paths[idx], exist_ok=True)
                
                with open(os.path.join(model_paths[idx], "loss.txt"), "a") as f:
                    msg = 'Epoch: %d | Training loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f'\
                             % (epoch, loss/(batch_idx+1), 100.*correct/total, correct, total, \
                                DC_results_total[0]/(batch_idx+1), DC_results_total[1]/(batch_idx+1)) + "\n"
                    f.write(msg)
            batch_idx += 1



def test(epoch):
    global best_acc
    current_acc = []
    for idx in range(args.num_nets):
        test_loss = 0
        correct = 0
        total = 0
        DC_results_total = np.zeros(args.num_nets-1)
        for batch_idx,(inputs,targets) in enumerate(test_gen):
                outputs, loss, DC_results = eval_nets(nets, idx, inputs, targets, criterion)
                loss = criterion.CE(outputs, targets)

                test_loss += loss.numpy()
                predicted = tf.math.argmax(outputs, axis=1)
                targets = tf.math.argmax(targets, axis=1)

                total += len(targets)
                correct += tf.math.count_nonzero(tf.math.equal(predicted, (targets))).numpy()
                
                DC_results_total += DC_results

                progress_bar(batch_idx, len(test_gen), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, 
                            DC_results_total[0]/(batch_idx+1), DC_results_total[1]/(batch_idx+1)))
                
                #log loss
                if args.log_loss and batch_idx == len(test_gen)-1:
                    os.makedirs(model_paths[idx], exist_ok=True)

                    with open(os.path.join(model_paths[idx], "loss.txt"), "a") as f:
                        msg = 'Epoch: %d | Test loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f'\
                             % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, \
                                DC_results_total[0]/(batch_idx+1), DC_results_total[1]/(batch_idx+1)) + "\n"
                        if idx == args.num_nets - 1:
                            msg = msg + "\n"
                        f.write(msg)
                
                batch_idx += 1

        # Save checkpoint.
        acc = 100.*correct/total
        current_acc.append(acc)

    if sum(current_acc) > sum(best_acc):
        print('Saving..')
        for idx in range(args.num_nets):
            os.makedirs(model_paths[idx], exist_ok=True)

            nets[idx].save(model_paths[idx])
            best_acc[idx] = current_acc[idx]
            checkpoints[idx].save(file_prefix=optim_paths[idx])
            
for epoch in range(start_epoch, args.epochs):
    train(epoch)
    test(epoch)

