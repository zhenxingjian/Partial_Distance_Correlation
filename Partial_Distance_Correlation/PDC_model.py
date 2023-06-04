import argparse
import numpy as np
import time
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from ViT_model import *
from resnet import *
from VGG import *
from densenet import *
from Partial_DC_grad import *
from utils import *

class PDC_Model(nn.Module):
    def __init__(self, modelX, modelY, normalize_X, normalize_Y):
        super(PDC_Model, self).__init__()
        self.modelX = modelX
        self.modelY = modelY
        self.normalize_X = normalize_X
        self.normalize_Y = normalize_Y

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputsX = self.normalize_X(inputs)
        featuresX = self.modelX.forward_features(inputsX)
        featuresX = featuresX.reshape([batch_size, -1])

        inputsY = self.normalize_Y(inputs)
        featuresY = self.modelY.forward_features(inputsY)
        # featuresY = self.modelY.global_pool(featuresY)
        featuresY = featuresY.reshape([batch_size, -1])

        matrix_A = P_Distance_Matrix(featuresX)
        matrix_B = P_Distance_Matrix(featuresY)

        matrix_A_B = P_removal(matrix_A, matrix_B)

        return matrix_A_B


class Model2Matrix(nn.Module):
    def __init__(self, modelX, normalize_X):
        super(Model2Matrix, self).__init__()
        self.modelX = modelX
        self.normalize_X = normalize_X

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputsX = self.normalize_X(inputs)
        featuresX = self.modelX.forward_features(inputsX)
        featuresX = featuresX.reshape([batch_size, -1])

        matrix_A = P_Distance_Matrix(featuresX)

        return matrix_A