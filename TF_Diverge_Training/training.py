"""Train CIFAR-10 with TensorFlow2.0."""
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
from DC_criterion import *
from models import *
from utils import *
from datetime import datetime
import matplotlib.pyplot as plt

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

#Modify your own model in models by 
#1. Returning the last feature output before softmax layer at your convenience
#2. Adapt first conv layer by input_shape (3*3 for cifar and 7*7 for ImageNet)  
#and add to this list
implemented_nets = ["resnet18", "resnet34", "resnet101", "resnet152"]     

parser = argparse.ArgumentParser(description='TensorFlow2.0 CIFAR-10 Training')
parser.add_argument('--num_nets', default=3, type=int, help='number of sub-networks')
parser.add_argument('--network', default='resnet18', type=str, help='model type', choices=implemented_nets)
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='number of training epoch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='specify which gpu to be used')
parser.add_argument("--notes", default=formatted_time, type=str, help="notes you want to add to the convergence plot")
parser.add_argument('--alpha', default=0.05, type=float, help='balance between accuracy and DC')
parser.add_argument('--dataset',type=str,help="cifar10 or imagenet",default="cifar10")
parser.add_argument("--aug", action="store_true", help="use data augmentation or not")
parser.add_argument("--debug", action="store_true", help="debug mode")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.network = args.network.lower()
CE_loss = tf.keras.losses.CategoricalCrossentropy()
class StackedTrainer():
    def __init__(self, model_type, scheduler, input_shape, num_classes, **kwargs):
        self.models = []
        for _ in range(args.num_nets):
            self.models.append(model_hanlder(model_type, input_shape, num_classes, return_feats=True))

        self.loss_func = Loss_DC(alpha=args.alpha)


        self.optims = [tf.keras.optimizers.SGD(learning_rate=scheduler[i], momentum=0.9) for i in range(len(self.models))]
        self.weight_decay = 5e-4
        self.train_loss = [tf.keras.metrics.Mean(name='train_loss') for _ in range(len(self.models))]
        self.train_acc = [tf.keras.metrics.CategoricalAccuracy(name='train_accuracy') for _ in range(len(self.models))]
        self.test_loss = [tf.keras.metrics.Mean(name='test_loss') for _ in range(len(self.models))]
        self.test_acc = [tf.keras.metrics.CategoricalAccuracy(name='test_accuracy') for _ in range(len(self.models))]
        self.update_funcs = None
        self.train_acc_to_plot = [[]] * len(self.models)
        self.test_acc_to_plot = [[]] * len(self.models)
        
    @tf.function
    def train_step(self, images, labels, net_idx):
        with tf.GradientTape() as tape:
            # Cross-entropy loss
            weights = self.models[net_idx].trainable_variables
            if args.debug:
                outputs = self.models[net_idx](images, training=True)
                main_loss = CE_loss(labels, outputs)
                DC_results = 0
            else:
                outputs, main_loss, DC_results  = run_nets(self.models, net_idx, images, labels, self.loss_func)
            # L2 loss(weight decay)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weights])
            loss = main_loss + l2_loss * self.weight_decay
        grads = tape.gradient(loss, weights)


        #Init update functions for each optimizer.
        #See https://www.tensorflow.org/guide/function#creating_tfvariables
        if self.update_funcs is None:
            self.update_funcs = [tf.function(self.update).get_concrete_function(weights, grads, i) for i in range(len(self.models))]
        # self.optims.apply_gradients(zip(gradients, self.models.trainable_variables))
        self.update_funcs[net_idx](weights, grads, net_idx)
        self.train_loss[net_idx](loss)
        self.train_acc[net_idx](labels, outputs)
        return DC_results
    


    def update(self, weights, grads, net_idx):
        self.optims[net_idx].apply_gradients(zip(grads, weights))
        

    @tf.function
    def test_step(self, images, labels, net_idx):
        if args.debug:
            outputs = self.models[net_idx](images, training=False)
            t_loss = CE_loss(labels, outputs)
            DC_results = 0
        else:
            outputs, t_loss, DC_results = eval_nets(self.models, net_idx, images, labels, self.loss_func)
        
        self.test_loss[net_idx](t_loss)
        self.test_acc[net_idx](labels, outputs)
        return DC_results
    

    def train(self, train_gen, test_gen, epoch):
        best_acc = [tf.Variable(0.0) for _ in range(len(self.models))]
        curr_epoch = [tf.Variable(0) for _ in range(len(self.models))]  # start from epoch 0 or last checkpoint epoch
        self.ckpt_path = ['./checkpoints/{:s}/net{:d}'.format(args.network, i) for i in range(len(self.models))]
        ckpt = [tf.train.Checkpoint(curr_epoch=curr_epoch[i], best_acc=best_acc[i],
                                   optimizer=self.optims[i], model=self.models[i]) for i in range(len(self.models))]
        managers = [tf.train.CheckpointManager(ckpt[i], self.ckpt_path[i], max_to_keep=1) for i in range(len(self.models))]
        
        
        for net_idx in range(len(self.models)):
            if args.resume:
                # Load checkpoint.
                print('==> Resuming from checkpoint...')
                assert os.path.isdir(self.ckpt_path[net_idx]), 'Error: no checkpoint directory found!'
                # Restore the weights
                ckpt[net_idx].restore(managers[net_idx].latest_checkpoint)

        for epoch in range(int(curr_epoch[net_idx]), args.epoch):
            print(f"--------------------------------------------------------Epoch {epoch}-----------------------------------------------------------------------------")
            
            for net_idx in range(len(self.models)):
                # Reset the metrics at the start of the next epoch
                self.train_loss[net_idx].reset_states()
                self.train_acc[net_idx].reset_states()
                self.test_loss[net_idx].reset_states()
                self.test_acc[net_idx].reset_states()
    
                
                #train over batches
                DC_results = np.zeros(len(self.models) - 1)
                for batch_idx, (images, labels) in enumerate(train_gen):
                    DC_results += self.train_step(images, labels, net_idx)
                    progress_bar(batch_idx, len(train_gen), f'Training: net {net_idx} | ' + 'Loss: %.3f | Acc: %.3f%% | DC0: %.3f | DC1: %.3f|'
                        % (self.train_loss[net_idx].result(), self.train_acc[net_idx].result() * 100,
                            DC_results[0]/(batch_idx+1), DC_results[1]/(batch_idx+1)))
                

                #test over batches
                DC_results = np.zeros(len(self.models) - 1)    
                for batch_idx, (images, labels) in enumerate(test_gen):
                    DC_results += self.test_step(images, labels, net_idx)
                    progress_bar(batch_idx, len(test_gen), f'Val: net {net_idx} | ' + 'Loss: %.3f | Acc: %.3f%% | DC0: %.3f | DC1: %.3f|'
                        % (self.test_loss[net_idx].result(), self.test_acc[net_idx].result() * 100,
                            DC_results[0]/(batch_idx+1), DC_results[1]/(batch_idx+1)))
                
                
                self.train_acc_to_plot[net_idx].append(self.train_acc[net_idx].result())
                self.test_acc_to_plot[net_idx].append(self.test_acc[net_idx].result())

                # Save checkpoint
                if self.test_acc[net_idx].result() > best_acc[net_idx]:
                    print('Saving...')
                    if not os.path.isdir('./checkpoints/'):
                        os.mkdir('./checkpoints/')
                    if not os.path.isdir(self.ckpt_path[net_idx]):
                        os.mkdir(self.ckpt_path[net_idx])
                    best_acc[net_idx].assign(self.test_acc[net_idx].result())
                    curr_epoch[net_idx].assign(epoch + 1)
                    managers[net_idx].save()

        self.plot_history()

    def predict(self, pred_ds, best, net_idx):
        if best:
            ckpt = tf.train.Checkpoint(model=self.models[net_idx])
            manager = tf.train.CheckpointManager(ckpt, self.ckpt_path[net_idx], max_to_keep=1)
            
            # Load checkpoint
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(self.ckpt_path[net_idx]), 'Error: no checkpoint directory found!'
            ckpt.restore(manager.latest_checkpoint)
        
        self.test_acc[net_idx].reset_states()
        for images, labels in pred_ds:
            self.test_step(images, labels, net_idx)
        print ('Prediction Accuracy: {:.2f}%'.format(self.test_acc[net_idx].result()*100))
        
    
    
    def plot_history(self):
        assert hasattr(self, "ckpt_path"), "No checkpoint path found. Please train the model first."
        for net_idx in range(len(self.models)):
            # Plot training history
            plt.plot(self.train_acc_to_plot[net_idx], label='training')
            plt.plot(self.test_acc_to_plot[net_idx], label='validation')
            plt.legend()
            plt.title('Training history')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.savefig(os.path.join(self.ckpt_path[net_idx], f'{args.notes}.png'))



def main():
    # Data
    print('==> Preparing data...')
    if args.dataset=="cifar10":
        #train_gen,test_gen = prepare_cifar10(args.batch_size)
        input_shape = (32,32,3)
        train_gen, test_gen = prepare_cifar10(args.batch_size, args.aug) 
        scheduler = CosineDecay(
                                args.lr,
                                steps_per_epoch=len(train_gen),
                                decay_steps=args.epoch,
                                alpha=0.0,
                                name=None)#decay every epoch
        num_classes = 10
        
    elif args.dataset=="imagenet":
        input_shape = (224,224,3)
        train_gen,test_gen = prepare_imagenet(args.batch_size)
        #My customized MultiStepLR. Keras doesn't implement this.
        scheduler = MultiStepLR(
                                lr=args.lr,
                                steps_per_epoch=len(train_gen),
                                milestones=[10,20,30],
                                gamma=0.1
                                )
        num_classes = 1000           
    scheduler = [scheduler for _ in range(args.num_nets)] 
        
    # Train
    print('==> Building model...')
    models = StackedTrainer(args.network, scheduler, input_shape, num_classes)
    models.train(train_gen, test_gen, args.epoch)
    

    # Evaluate
    for net_idx in range(args.num_nets):
        models.predict(test_gen, best=True, net_idx=net_idx)

if __name__ == "__main__":
    main()
