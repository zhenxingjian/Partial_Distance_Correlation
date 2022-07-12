from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import numpy as np
import time
import os
import pdb
import cv2

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

from PDC_model import *
from PDC_CAM import *


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

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Partial Distance Correlation Grad-CAM')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                                  help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')

    args = parser.parse_args()

    modelX = ViT()


    # Strange loading method
    pretrained_dict = torch.load('./ViT_model/vit.pth')['net']
    pretrained_dictX = {}
    modelX_dict = modelX.state_dict()
    for k in pretrained_dict:
        if 'modelX' in k:
            new_key = '.'.join(k.split('.')[2:])
            pretrained_dictX[new_key] = pretrained_dict[k]
    modelX_dict.update(pretrained_dictX)
    modelX.load_state_dict(pretrained_dictX)

    old_modelX = ViT()


    normalize_X = get_normalize_layer(modelX.default_cfg['mean'], modelX.default_cfg['std'])
    modelY = resnet18()
    normalize_Y = get_normalize_layer(modelY.default_cfg['mean'], modelY.default_cfg['std'])

    test_dataset = get_dataset('test')

    pin_memory = True

    torch.manual_seed(5)
    test_loader = DataLoaderX(test_dataset, shuffle=True, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=pin_memory)


    num_classes = 1000

    model_remove = PDC_Model(modelX, modelY, normalize_X, normalize_Y)
    model_X_DC = Model2Matrix(old_modelX, normalize_X)
    model_Y_DC = Model2Matrix(modelY, normalize_Y)

    target_layers = [model_remove.modelX.blocks[-1].norm1]
    target_layers_X = [model_X_DC.modelX.blocks[-1].norm1]
    target_layers_Y = [model_Y_DC.modelX.layer4[-1]]


    ImageNetLabelEmbedding = torch.load('ImageNet_Class_Embedding.pt')
    ImageNetLabelEmbedding = ImageNetLabelEmbedding.to(device)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        input_tensor = inputs.cuda()

        cam = PDC_CAM(model=model_remove, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)

        target_category = ImageNetLabelEmbedding[targets]

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)


        
        cam_X = PDC_CAM(model=model_X_DC, target_layers=target_layers_X, use_cuda=True, reshape_transform=reshape_transform)

        grayscale_cam_X = cam_X(input_tensor=input_tensor, target_category=target_category)


        cam_Y = PDC_CAM(model=model_Y_DC, target_layers=target_layers_Y, use_cuda=True)

        grayscale_cam_Y = cam_Y(input_tensor=input_tensor, target_category=target_category)

        if not os.path.exists('./result/'):
            os.mkdir('./result/')
        for idx in range(len(targets)):
            grayscale_cam_idx = grayscale_cam[idx, :]
            rgb_img = inputs.permute(0,2,3,1).numpy()
            rgb_img = rgb_img[idx, :]
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            cv2.imwrite('./result/'+str(idx)+f'_XY_cam_ori.jpg', rgb_img*255)
            visualization = show_cam_on_image(rgb_img, grayscale_cam_idx, use_rgb=False)
            cv2.imwrite('./result/'+str(idx)+f'_XY_cam.jpg', visualization)

            grayscale_cam_X_idx = grayscale_cam_X[idx, :]

            visualization_X = show_cam_on_image(rgb_img, grayscale_cam_X_idx, use_rgb=False)
            cv2.imwrite('./result/'+str(idx)+f'_X_cam.jpg', visualization_X)

            grayscale_cam_Y_idx = grayscale_cam_Y[idx, :]
            visualization_Y = show_cam_on_image(rgb_img, grayscale_cam_Y_idx, use_rgb=False)

            cv2.imwrite('./result/'+str(idx)+f'_Y_cam.jpg', visualization_Y)

        break
