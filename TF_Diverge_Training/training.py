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
from models.resnet import *
import tensorflow.keras.layers as layers

#add to this list if you implement more models in models.py
implemented_nets = ["resnet18", "resnet34", "resnet101", "resnet152"]     

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.05, type=float, help='balance between accuracy and DC')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_nets', default=3, type=int, help='number of sub-networks')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# parser.add_argument('--workers', default=2, type=int, help='number of workers for dataloader')
parser.add_argument('--network', default='resnet18', type=str, help='name of the network', choices=implemented_nets)
parser.add_argument('--epochs',type=int,help="training epochs. 200 according to the paper.",default=200)
parser.add_argument('--dataset',type=str,help="cifar10 or imagenet",default="cifar10")
parser.add_argument('--log_loss', "-l", type=eval, default=True, help='log training and validation loss in txt files')
parser.add_argument('--debug', action='store_true', help='debug mode using only CE loss')
parser.add_argument("--aug", action="store_true", help="use data augmentation or not")
parser.add_argument("--notes", type=str, help="notes for this run")
parser.add_argument("--gpu", type=str, help="gpu to use", default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.network = args.network.lower()

debug = args.debug #1. CosineDecay could cause the discrepancy 
return_feats = not debug


best_acc = [0 for _ in range(args.num_nets)]  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_acc = keras.metrics.CategoricalAccuracy()
test_acc = keras.metrics.CategoricalAccuracy()
train_loss = keras.metrics.Mean(name='train_loss')
test_loss = keras.metrics.Mean(name='test_loss')
train_acc_to_plot = []
test_acc_to_plot = []

# Get dataset
if args.dataset=="cifar10":
    #train_gen,test_gen = prepare_cifar10(args.batch_size)
    input_shape = (32,32,3)
    train_gen, test_gen = prepare_cifar10(args.batch_size, args.aug) 
    scheduler = CosineDecay(
                            args.lr,
                            steps_per_epoch=len(train_gen),
                            decay_steps=200,
                            alpha=0.0,
                            name=None)#decay every epoch

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
    raise NotImplementedError("%s is not implemented! Available options are " % args.network, implemented_nets)

if debug:
    nets = [model_name(num_classes=10) for _ in range(args.num_nets)]
else:
    if args.dataset=='cifar10':
        nets = [model_name(input_shape=(None, 32,32,3), num_classes=10, return_feats=return_feats) for _ in range(args.num_nets)]
    else:
        nets = [model_name(input_shape=(None, 224,224,3), num_classes=10, return_feats=return_feats) for _ in range(args.num_nets)]

        
try:
    optims = [keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9, decay=5e-4) for _ in range(args.num_nets)]
except:
    optims = [keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9, weight_decay=5e-4) for _ in range(args.num_nets)]
    
model_paths = [f'./checkpoints/{args.network}/ckpt_' + str(idx) + "_" + args.dataset for idx in range(args.num_nets) ]
optim_paths = [os.path.join(model_paths[idx], "optim") for idx in range(args.num_nets)]
checkpoints = [tf.train.Checkpoint(optmizer=optims[idx]) for idx in range(args.num_nets)]
checkpoints = [tf.train.CheckpointManager(checkpoints[idx], optim_paths[idx], max_to_keep=1) for idx in range(args.num_nets)]
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
    
    # @tf.function doesn't work for multiple optimizers
    def step(inputs, targets, idx):
        if debug:
            with tf.GradientTape() as tape:
                outputs = nets[idx](inputs, training=True)
                loss = criterion.ce(targets, outputs)
                grads = tape.gradient(loss, nets[idx].trainable_weights)
                DC_results = [0] * (args.num_nets-1)
                
        else:
            outputs, loss, DC_results = run_nets(nets, idx, inputs, targets, criterion)
    
        #This updates steps used for cosine decay
        optims[net_idx].apply_gradients(zip(grads, nets[net_idx].trainable_weights))
        train_acc(targets, outputs)
        train_loss(loss)

        return DC_results
    
    print('\nEpoch: %d' % epoch)
    for net_idx in range(args.num_nets):
        DC_results_total = np.zeros(args.num_nets-1)
        train_acc.reset_states()
        train_loss.reset_states()

        for batch_idx, (inputs ,targets) in enumerate(train_gen):
            
            DC_results = step(inputs, targets, net_idx)
            DC_results = [item.numpy() if isinstance(item, tf.Tensor) else item for item in DC_results ]
            DC_results_total += DC_results
            
            progress_bar(batch_idx, len(train_gen), 'Loss: %.3f | Acc: %.3f%% | DC0: %.3f | DC1: %.3f'
                        % (train_loss.result(), train_acc.result() * 100,
                            DC_results_total[0] / (batch_idx + 1), 
                            DC_results_total[1] / (batch_idx + 1))
                            )
            train_acc_to_plot.append(train_acc.result())

            #log loss 
            if args.log_loss and batch_idx == len(train_gen)-1:
                os.makedirs(model_paths[net_idx], exist_ok=True)
                
                with open(os.path.join(model_paths[net_idx], "loss.txt"), "a") as f:
                    msg = f"Epoch: {epoch} | Loss: {train_loss.result()} | Acc: {train_acc.result() * 100} | DC0: {DC_results_total[0]/(batch_idx+1)} | DC1: {DC_results_total[1]/(batch_idx+1)}\n"
                    f.write(msg)
            batch_idx += 1


def test(epoch):
    @tf.function
    def step(inputs, targets, idx):
        if debug:
            outputs = nets[idx](inputs, training=False)
            loss = criterion.ce(targets, outputs)
            DC_results = [0] * (args.num_nets-1)
        else:
            outputs, loss, DC_results = eval_nets(nets, idx, inputs, targets, criterion)

        test_acc(targets, outputs)
        test_loss(loss)
        return DC_results
    
    global best_acc
    for idx in range(args.num_nets):
        DC_results_total = np.zeros(args.num_nets-1)
        test_acc.reset_states()
        test_loss.reset_states()
        
        for batch_idx,(inputs,targets) in enumerate(test_gen):
                DC_results = step(inputs, targets, idx)
                DC_results = [item.numpy() if isinstance(item, tf.Tensor) else item for item in DC_results]
                DC_results_total += DC_results

                progress_bar(batch_idx, len(test_gen), 'Loss: %.3f | Acc: %.3f%% | DC0: %.3f | DC1: %.3f'
                            % (test_loss.result(), test_acc.result() * 100,
                                DC_results_total[0] / (batch_idx + 1),
                                DC_results_total[1] / (batch_idx + 1))
                )
                
                #log loss
                if args.log_loss and batch_idx == len(test_gen)-1:
                    os.makedirs(model_paths[idx], exist_ok=True)

                    with open(os.path.join(model_paths[idx], "loss.txt"), "a") as f:
                        msg = 'Epoch: %d | Test loss: %.3f | Acc: %.3f%% | DC0: %.3f | DC1: %.3f'\
                             % (epoch, test_loss.result(), test_acc.result() * 100, DC_results_total[0]/(batch_idx+1), DC_results_total[1]/(batch_idx+1))
                        
                        msg = msg + "\n" # end of epoch
                        f.write(msg)
                
                batch_idx += 1

        # Save checkpoint.
        test_acc_to_plot.append(test_acc.result().numpy() * 100)
        
    if sum(test_acc_to_plot) > sum(best_acc):
        print('Saving..')
        for idx in range(args.num_nets):
            os.makedirs(model_paths[idx], exist_ok=True)

            nets[idx].save(model_paths[idx])
            best_acc[idx] = test_acc_to_plot[idx]
            checkpoints[idx].save()
            
for epoch in range(start_epoch, args.epochs):
    train(epoch)
    test(epoch)
    
#plot training history
import matplotlib.pyplot as plt
plt.plot(train_acc_to_plot, label="train_acc")
plt.plot(test_acc_to_plot, label="test_acc")
plt.legend()
plt.title("Training history")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(f"{args.notes}_training_history.png")