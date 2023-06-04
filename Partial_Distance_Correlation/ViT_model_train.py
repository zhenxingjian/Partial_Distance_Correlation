import timm
import torch
import argparse
import numpy as np
import time
import os
import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from Partial_DC_grad import *

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


class ViT_train(nn.Module):
    def __init__(self):
        super(ViT_train, self).__init__()
        self.modelX = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.modelY = timm.create_model('resnet18', pretrained=True)
        self.normalize_X = NormalizeLayer(self.modelX.default_cfg['mean'], self.modelX.default_cfg['std'])
        self.normalize_Y = NormalizeLayer(self.modelY.default_cfg['mean'], self.modelY.default_cfg['std'])

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputsX = self.normalize_X(inputs)
        featuresX = self.modelX.forward_features(inputsX)

        outputs = self.modelX.head(featuresX)

        featuresX = featuresX.reshape([batch_size, -1])

        inputsY = self.normalize_Y(inputs)
        featuresY = self.modelY.forward_features(inputsY)
        # featuresY = self.modelY.global_pool(featuresY)
        featuresY = featuresY.reshape([batch_size, -1])

        # matrix_A = P_Distance_Matrix(featuresX)
        # matrix_B = P_Distance_Matrix(featuresY)

        # matrix_A_B = P_removal(matrix_A, matrix_B)

        return outputs, featuresX, featuresY
