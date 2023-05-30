import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch.nn.init as init

import torchvision.transforms as transforms



def run_nets(net, idx, inputs, targets, criterion, args):
    eval_sub_net = list(range(idx))
    for sub_net_idx in eval_sub_net:
        net[sub_net_idx].eval()
    net[idx].train()
    ref_features = []
    for sub_net_idx in eval_sub_net:
        _, feature = net[sub_net_idx](inputs)
        ref_features.append(feature.detach())

    outputs, learned_feature = net[idx](inputs)
    loss, _, _, DC_results = criterion(outputs, targets, learned_feature, ref_features)
    if len(DC_results) < args.num_nets - 1:
        for _ in range(args.num_nets - 1 - len(DC_results)):
            DC_results.append(0.0)
    DC_results = np.asarray(DC_results)
    return outputs, loss, DC_results


def eval_nets(net, idx, inputs, targets, criterion, args):
    eval_sub_net = list(range(idx))
    for sub_net_idx in eval_sub_net:
        net[sub_net_idx].eval()
    net[idx].eval()
    ref_features = []
    for sub_net_idx in eval_sub_net:
        _, feature = net[sub_net_idx](inputs)
        ref_features.append(feature.detach())

    outputs, learned_feature = net[idx](inputs)
    loss, _, _, DC_results = criterion(outputs, targets, learned_feature, ref_features)
    if len(DC_results) < args.num_nets - 1:
        for _ in range(args.num_nets - 1 - len(DC_results)):
            DC_results.append(0.0)
    DC_results = np.asarray(DC_results)
    return outputs, loss, DC_results


class Loss_DC(nn.Module):
    def __init__(self, alpha = 0.1):
        super(Loss_DC, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        print("Loss balance alpha is: ", alpha)

    def CE(self, logit, target):
        return self.ce(logit, target)


    def Distance_Correlation(self, latent, control):

        latent = F.normalize(latent)
        control = F.normalize(control)

        matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
        matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)

        matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
        matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

        Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

        
        correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        # correlation_r = torch.pow(Gamma_XY,2)/(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r


    def forward(self, logit, target, latent, controls):
        # cls_loss = self.BCE(logit, target)
        cls_loss = self.CE(logit, target)
        dc_loss = 0
        DC_results = []
        for control in controls:
            DC = self.Distance_Correlation(latent, control)
            dc_loss += DC
            DC_results.append(DC.detach().cpu().item())

        dc_loss /= (len(controls) + 1e-12)
        #dc_loss *= logit.shape[0]

        loss = cls_loss + self.alpha * dc_loss
        return loss, cls_loss, dc_loss, DC_results
