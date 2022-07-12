import argparse
import numpy as np
import time
import os
import pdb

import torch
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



def eval_model(modelX, modelY, testloader, normalize_X, normalize_Y):
    modelX.eval()
    modelY.eval()
    correctX = 0
    correctY = 0
    total = 0  
    total_pdv = 0
    total_pdv_inv = 0
    total_dcX = 0
    total_dcY = 0

    ImageNetLabelEmbedding = torch.load('ImageNet_Class_Embedding.pt')
    ImageNetLabelEmbedding = ImageNetLabelEmbedding.to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            batch_size = inputs.shape[0]

            inputsX = normalize_X(inputs)
            outputsX = modelX(inputsX)
            featuresX = modelX.module.forward_features(inputsX)
            featuresX = featuresX.reshape([batch_size, -1])
            
            inputsY = normalize_Y(inputs)
            outputsY = modelY(inputsY)
            featuresY = modelY.module.forward_features(inputsY)
            featuresY = featuresY.reshape([batch_size, -1])

            gt_embedding = ImageNetLabelEmbedding[targets]

            pdcov = P_DC(featuresX, featuresY, gt_embedding)
            pdcov_inv = P_DC(featuresY, featuresX, gt_embedding)
            dcX = New_DC(featuresX, gt_embedding)
            dcY = New_DC(featuresY, gt_embedding)

            total_pdv += pdcov
            total_pdv_inv += pdcov_inv
            total_dcX += dcX
            total_dcY += dcY

            _, predictedX = outputsX.max(1)

            _, predictedY = outputsY.max(1)

            total += targets.size(0)


            correctX += predictedX.eq(targets).sum().item()
            correctY += predictedY.eq(targets).sum().item()


            progress_bar(batch_idx, len(testloader), 'Acc X: %.3f%% (%d/%d) | Acc Y: %.3f%% (%d/%d) | pd cor: %.3f | pd cor inv: %.3f | corX: %.3f | corY: %.3f'
                         % (100.*correctX/total, correctX, total, 100.*correctY/total, correctY, total, 
                            total_pdv/(batch_idx + 1), total_pdv_inv/(batch_idx + 1), total_dcX/(batch_idx + 1), total_dcY/(batch_idx + 1) ))




def main():

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Partial Distance Correlation')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                                  help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')

    args = parser.parse_args()

    modelX = resnet50()
    normalize_X = get_normalize_layer(modelX.default_cfg['mean'], modelX.default_cfg['std'])
    modelY = VGG19_BN()
    normalize_Y = get_normalize_layer(modelY.default_cfg['mean'], modelY.default_cfg['std'])

    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')

    pin_memory = True

    train_loader = DataLoaderX(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoaderX(test_dataset, shuffle=True, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=pin_memory)


    num_classes = 1000

    modelX.to(device)
    modelY.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
        modelX = torch.nn.DataParallel(modelX)
        modelY = torch.nn.DataParallel(modelY)


    eval_model(modelX, modelY, test_loader, normalize_X, normalize_Y)



