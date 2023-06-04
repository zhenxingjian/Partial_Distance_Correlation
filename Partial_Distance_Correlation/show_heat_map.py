import argparse
import numpy as np
import time
import os
import pdb

from pathlib import Path
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch 
from torchmetrics import Metric, BootStrapper
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# plt.style.use('ggplot')


if __name__ == '__main__':
    PATH = './ViT_resnet34'

    sim_mat = np.load(PATH + 'modelX_model_X.npy')

    figure(figsize=(6,6), dpi=60)
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('ViT-B/16', fontsize=50)
    plt.xlabel('Layers ViT-B/16', fontsize=30)
    plt.ylabel('Layers ViT-B/16', fontsize=30)
    plt.savefig('ViT.png', dpi=300)
    plt.show()

    figure(figsize=(6,6), dpi=60)
    sim_mat = np.load(PATH + 'modelY_model_Y.npy')
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('Resnet 34', fontsize=50)
    plt.xlabel('Layers Resnet 34', fontsize=30)
    plt.ylabel('Layers Resnet 34', fontsize=30)
    plt.savefig('Resnet34.png', dpi=300)
    plt.show()
    
    figure(figsize=(9,12), dpi=60)
    sim_mat = np.load(PATH + 'modelX_model_Y.npy')
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('ViT-B/16 vs R34', fontsize=75)
    plt.ylabel('Layers ViT-B/16', fontsize=65)
    plt.xlabel('Layers Resnet 34', fontsize=65)
    plt.savefig('ViT_Resnet34.png', dpi=300)
    plt.show()


    PATH = './ViT_resnet50'
    figure(figsize=(6,6), dpi=60)
    sim_mat = np.load(PATH + 'modelY_model_Y.npy')
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('Resnet 50', fontsize=50)
    plt.xlabel('Layers Resnet 50', fontsize=30)
    plt.ylabel('Layers Resnet 50', fontsize=30)
    plt.savefig('Resnet50.png', dpi=300)
    plt.show()
    
    figure(figsize=(12,12), dpi=60)
    sim_mat = np.load(PATH + 'modelX_model_Y.npy')
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('ViT-B/16 vs R50', fontsize=75)
    plt.ylabel('Layers ViT-B/16', fontsize=65)
    plt.xlabel('Layers Resnet 50', fontsize=65)
    plt.savefig('ViT_Resnet50.png', dpi=300)
    plt.show()


    PATH = './ViT_resnet152'
    figure(figsize=(7,6), dpi=60)
    sim_mat = np.load(PATH + 'modelY_model_Y.npy')
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('Resnet 152', fontsize=50)
    plt.xlabel('Layers Resnet 152', fontsize=30)
    plt.ylabel('Layers Resnet 152', fontsize=30)
    plt.savefig('Resnet152.png', dpi=300)
    plt.show()
    
    figure(figsize=(30,12), dpi=60)
    sim_mat = np.load(PATH + 'modelX_model_Y.npy')
    plt.imshow(sim_mat, origin='lower', cmap="plasma")
    plt.title('ViT-B/16 vs R152', fontsize=75)
    plt.ylabel('Layers ViT-B/16', fontsize=65)
    plt.xlabel('Layers Resnet 152', fontsize=65)
    plt.savefig('ViT_Resnet152.png', dpi=300)
    plt.show()
    