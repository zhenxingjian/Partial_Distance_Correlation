import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import os
import argparse

from DC_criterion import *
from utils import progress_bar
import torchvision.models as models
from resnet import resnet152, resnet34

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.05, type=float, help='balance between accuracy and DC')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_nets', default=3, type=int, help='number of sub-networks')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--workers', default=2, type=int, help='number of workers for dataloader')
parser.add_argument('--network', default='resnet152', type=str, help='name of the network')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = [0 for _ in range(args.num_nets)]  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
'''
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)
'''
# debug
# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

### ImageNet loader
IMAGENET_LOC_ENV = "/nobackup/imagenet/2012/"
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
def _imagenet(split):

    dir = IMAGENET_LOC_ENV
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])       
        #transform = transforms.Compose([
        #    transforms.RandomResizedCrop(224),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    normalize,
        #])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return datasets.ImageFolder(subdir, transform)
def get_dataset(split):
    """Return the dataset as a PyTorch Dataset object"""
    return _imagenet(split)
train_dataset = get_dataset('train')
test_dataset = get_dataset('test')

pin_memory = True
trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=pin_memory)
testloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=pin_memory)

# Model
print('==> Building model..')
#net = ResNet18(num_nets = args.num_nets)
if args.network == 'resnet152':
    model_name = resnet152
elif args.network == 'resnet34':
    model_name = resnet34
elif args.network == 'efficientnet':
    model_name = models.efficientnet_b0
elif args.network == 'mobilenet':
    model_name = models.mobilenet_v3_small
net = [model_name() for _ in range(args.num_nets)]

net = [ subnet.to(device) for subnet in net ]
if device == 'cuda':
    net = [torch.nn.DataParallel(subnet) for subnet in net]
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists('./checkpoint/ckpt_'+str(args.num_nets-1)+'.pth')
    for idx in range(args.num_nets):
        checkpoint = torch.load('./checkpoint/ckpt_'+str(idx)+'.pth')
        net[idx].load_state_dict(checkpoint['net'])
        best_acc[idx] = checkpoint['acc']
        start_epoch = max(start_epoch, checkpoint['epoch'])

# TODO: Change it into DC version
# criterion = nn.CrossEntropyLoss() 
criterion = Loss_DC(alpha = args.alpha)

optimizers = []
schedulers = []
for idx in range(args.num_nets):
    optimizer = optim.SGD(net[idx].parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    optimizers.append(optimizer)

    #schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120))
    schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
          optimizer, milestones=[10, 20, 30], gamma=0.1))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    for idx in range(args.num_nets):
        train_loss = 0
        correct = 0
        total = 0

        DC_results_total = np.zeros(args.num_nets-1)

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizers[idx].zero_grad()
            # outputs = net[idx](inputs)
            # loss = criterion(outputs, targets)
            outputs, loss, DC_results = run_nets(net, idx, inputs, targets, criterion, args)
            loss.backward()
            optimizers[idx].step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            DC_results_total += DC_results

            # print(DC_results)

            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f | DC2: %.3f | DC3: %.3f'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, 
            #                DC_results_total[0]/(batch_idx+1) , DC_results_total[1]/(batch_idx+1) , 
            #                DC_results_total[2]/(batch_idx+1) , DC_results_total[3]/(batch_idx+1) ))
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, 
                            DC_results_total[0]/(batch_idx+1) , DC_results_total[1]/(batch_idx+1) ))



def test(epoch):
    global best_acc
    current_acc = []
    for idx in range(args.num_nets):
        net[idx].eval()
        test_loss = 0
        correct = 0
        total = 0
        DC_results_total = np.zeros(args.num_nets-1)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs, loss, DC_results = eval_nets(net, idx, inputs, targets, criterion, args)
                # outputs, _ = net[idx](inputs)
                loss = criterion.CE(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                DC_results_total += DC_results

                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f | DC2: %.3f | DC3: %.3f'
                #         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, 
                #            DC_results_total[0]/(batch_idx+1) , DC_results_total[1]/(batch_idx+1) , 
                #            DC_results_total[2]/(batch_idx+1) , DC_results_total[3]/(batch_idx+1) ))
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | DC0: %.3f | DC1: %.3f'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, 
                            DC_results_total[0]/(batch_idx+1) , DC_results_total[1]/(batch_idx+1) ))


        # Save checkpoint.
        acc = 100.*correct/total
        current_acc.append(acc)
    if sum(current_acc) > sum(best_acc):
        print('Saving..')
        for idx in range(args.num_nets):
            state = {
                'net': net[idx].state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/%s_ckpt_'%args.network+str(idx)+'.pth')
            best_acc[idx] = current_acc[idx]


for epoch in range(start_epoch, start_epoch+40):
    #test(epoch)
    train(epoch)
    test(epoch)
    for scheduler in schedulers:
        scheduler.step()
