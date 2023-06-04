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

from ViT_model_train import *
from Partial_DC_grad import *
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



def ImageNet_train(model, trainloader, optimizer, dcloss, device):
    ce_total = 0
    dc_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs, featureX, featureY = model(inputs)

        # breakpoint()

        loss, loss_CE, loss_DC = dcloss(outputs, featureX, featureY, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_total += loss_CE.item()
        dc_total += loss_DC.item()

        progress_bar(batch_idx, len(trainloader), 'CE loss: %.3f | DC: %.3f'
                         % (ce_total/(batch_idx+1), dc_total/(batch_idx+1) ))
    print('Classification Loss: {}  DC: {}'.format(ce_total/(batch_idx+1), dc_total/(batch_idx+1)))


def ImageNet_test(model, testloader, dcloss, device):
    model.eval()
    ce_total = 0
    dc_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs,  featureX, featureY = model(inputs)

            loss, loss_CE, loss_DC = dcloss(outputs, featureX, featureY, targets)

            ce_total += loss_CE.item()
            dc_total += loss_DC.item()
            progress_bar(batch_idx, len(testloader), 'CE loss: %.3f | DC: %.3f'
                         % (ce_total/(batch_idx+1), dc_total/(batch_idx+1) ))
    print('Classification Loss: {}  DC: {}'.format(ce_total/(batch_idx+1), dc_total/(batch_idx+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Partial Distance Correlation')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                                  help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')

    parser.add_argument('--lr', default=0.00001, type=float, help='Initial learning rate')

    args = parser.parse_args()

    ckptdir = './ViT_model/'
    if not os.path.exists(ckptdir):
        os.mkdir(ckptdir)

    model = ViT_train()
    
    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')

    pin_memory = True

    train_loader = DataLoaderX(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoaderX(test_dataset, shuffle=True, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=pin_memory)


    num_classes = 1000

    model.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(
          model.module.modelX.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    dcloss = Loss_DC(1).to(device)

    scheduler = MultiStepLR(
          optimizer, milestones=[30, 60, 90], gamma=0.1)

    for epoch in range(90):
        ImageNet_train(model, train_loader, optimizer, dcloss, device)

        if epoch%5 == 0:
            ImageNet_test(model, test_loader, dcloss, device)

        if ckptdir is not None:
            # Save checkpoint
            print('==> Saving {}.pth..'.format(epoch))
            try:
                state = {
                      'net': model.state_dict(),
                      'epoch': epoch,
                }
                torch.save(state, '{}/{}.pth'.format(ckptdir, epoch))
            except OSError:
                print('OSError while saving {}.pth'.format(epoch))
                print('Ignoring...')



