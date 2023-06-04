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
import timm

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from ViT_model import *
from resnet import *
from VGG import *
from densenet import *
from Partial_DC import *
from utils import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGENET_LOC_ENV = "/PATH/ImageNet"


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_dataset(split):
    """Return the dataset as a PyTorch Dataset object"""
    return _imagenet(split)

def get_num_classes():
    """Return the number of classes in the dataset. """
    return 1000


def get_normalize_layer(_IMAGENET_MEAN, _IMAGENET_STDDEV):
    return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)

def _imagenet(split):

    dir = IMAGENET_LOC_ENV
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])
    return datasets.ImageFolder(subdir, transform)


class HookedCache:
    def __init__(self, model, target):
        self.model = model
        self.target = target
        
        self.clear()
        self._extract_target()
        self._register_hook()

    @property
    def value(self):
        return self._cache
    def clear(self):
        self._cache = None
    def _extract_target(self):
        for name, module in self.model.named_modules():
          if name == self.target:
              self._target = module
              return
    def _register_hook(self):
        def _hook(module, in_val, out_val):
             self._cache = out_val
        self._target.register_forward_hook(_hook)


class NormalizeLayer(nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """
    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.nn.parameter.Parameter(torch.tensor(means))
        self.sds = torch.nn.parameter.Parameter(torch.tensor(sds))
        self.means.requires_grad = False
        self.sds.requires_grad = False

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


class MinibatchDC(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Introduced in: https://arxiv.org/pdf/2010.15327.pdf
        Implemented to reproduce the results in: https://arxiv.org/pdf/2108.08810v1.pdf
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("_cr", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, X: torch.Tensor, Y: torch.Tensor):
        self._cr += New_DC(X,Y)
    def compute(self):
        return self._cr


def make_pairwise_metrics(mod1_hooks, mod2_hooks):
    metrics = []
    for i_ in mod1_hooks:
        metrics.append([])
        for j_ in mod2_hooks:
            metrics[-1].append(MinibatchDC().to(device))
    return metrics

def update_metrics(mod1_hooks, mod2_hooks, metrics):
    for i, hook1 in enumerate(mod1_hooks):
      for j, hook2 in enumerate(mod2_hooks):
        DC = metrics[i][j]
        X,Y = hook1.value, hook2.value
        DC.update(X,Y)
        

def get_simmat_from_metrics(metrics):
    vals = []
    for i, dcs in enumerate(metrics):
      for j, dc in enumerate(dcs):
        z = dc.compute().item()
        vals.append((i,j,z))

    sim_mat = torch.zeros(i+1,j+1)
    for i,j,z in vals:
      sim_mat[i,j] = z
    
    return sim_mat



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Heat Map Plot')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                                  help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--PATH', default='./ViT_resnet34', type=str, help='Path to save the heat map results')
    parser.add_argument('--total_time', default=3600.0, type=float, help='longest running time')

    args = parser.parse_args()
    
    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')

    pin_memory = True

    train_loader = DataLoaderX(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoaderX(test_dataset, shuffle=True, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=pin_memory)


    num_classes = 1000

    modelX = ViT()
    modelY = resnet152()

    normalize_X = NormalizeLayer(modelX.default_cfg['mean'], modelX.default_cfg['std'])
    normalize_Y = NormalizeLayer(modelY.default_cfg['mean'], modelY.default_cfg['std'])

    modelX.to(device)
    modelY.to(device)
    
        
    
    each_blocks = ['.norm1', '.attn', '.norm2', '.mlp.fc1', '.mlp.fc2']
    modelX_hooks = []

    hook = HookedCache(modelX, 'patch_embed')
    modelX_hooks.append(hook)
    hook = HookedCache(modelX, 'pos_drop')
    modelX_hooks.append(hook)

    for j, block in enumerate(modelX.blocks):
        tgt = f'blocks.{j}'
        for b_ in each_blocks:
            hook = HookedCache(modelX, tgt+b_)
            modelX_hooks.append(hook)

    hook = HookedCache(modelX, 'norm')
    modelX_hooks.append(hook)


    each_block_Y = ['.conv1', '.conv2', '.conv3']
    modelY_hooks = []

    for name, layer in modelY.named_children():
        if 'conv1' in name or 'fc' in name:
            tgt = name
            hook = HookedCache(modelY, tgt)
            modelY_hooks.append(hook)
        elif 'layer' in name:
            for subname, _ in layer.named_children():
                for b_ in each_block_Y:
                    tgt = name + '.' + subname + b_
                    try:
                        hook = HookedCache(modelY, tgt)
                        modelY_hooks.append(hook)
                    except:
                        print(tgt+' not in modelY')


    metrics_XY = make_pairwise_metrics(modelX_hooks, modelY_hooks)
    metrics_XX = make_pairwise_metrics(modelX_hooks, modelX_hooks)
    metrics_YY = make_pairwise_metrics(modelY_hooks, modelY_hooks)

    starting_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            tic = time.time()
            inputs, targets = inputs.to(device), targets.to(device)

            outv_c = modelX(inputs)
            outv_t = modelY(inputs)

            update_metrics(modelX_hooks, modelY_hooks, metrics_XY)
            update_metrics(modelX_hooks, modelX_hooks, metrics_XX)
            update_metrics(modelY_hooks, modelY_hooks, metrics_YY)

            for hook0 in modelX_hooks:
                for hook1 in modelY_hooks:
                    hook0.clear()
                    hook1.clear()

            print(batch_idx, '/', len(test_loader), ' | time: ', time.time()-tic)

            if time.time() - starting_time > args.total_time:
                break

    sim_mat = get_simmat_from_metrics(metrics_XX)
    np.save(args.PATH + 'modelX_model_X.npy', sim_mat.numpy())

    sim_mat = get_simmat_from_metrics(metrics_YY)
    np.save(args.PATH + 'modelY_model_Y.npy', sim_mat.numpy())

    sim_mat = get_simmat_from_metrics(metrics_XY)
    np.save(args.PATH + 'modelX_model_Y.npy', sim_mat.numpy())


