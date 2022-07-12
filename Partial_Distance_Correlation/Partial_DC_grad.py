import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def Distance_Correlation(latent, control):

    latent = F.normalize(latent)
    control = F.normalize(control)

    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)

    matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

    
    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    # correlation_r = torch.pow(Gamma_XY,2)/(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r

def P_Distance_Matrix(latent):
    n = latent.shape[0]
    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1)  + 1e-18)
    matrix_A = matrix_a - torch.sum(matrix_a, dim = 0, keepdims= True)/(n-2) - torch.sum(matrix_a, dim = 1, keepdims= True)/(n-2) \
                + torch.sum(matrix_a)/((n-1)*(n-2))

    diag_A = torch.diag(torch.diag(matrix_A) ) 
    matrix_A = matrix_A - diag_A
    return matrix_A


def bracket_op(matrix_A, matrix_B):
    n = matrix_A.shape[0]
    return torch.sum(matrix_A * matrix_B)/(n*(n-3))


def P_removal(matrix_A, matrix_C):
    result = matrix_A - bracket_op(matrix_A, matrix_C) / bracket_op(matrix_C, matrix_C) * matrix_C
    return result

def Correlation(matrix_A, matrix_B):
    Gamma_XY = bracket_op(matrix_A, matrix_B)
    Gamma_XX = bracket_op(matrix_A, matrix_A)
    Gamma_YY = bracket_op(matrix_B, matrix_B)

    
    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-18)

    return correlation_r


def P_DC(latent_A, latent_B, ground_truth):
    matrix_A = P_Distance_Matrix(latent_A)
    matrix_B = P_Distance_Matrix(latent_B)
    matrix_GT = P_Distance_Matrix(ground_truth)

    # breakpoint()

    matrix_A_B = P_removal(matrix_A, matrix_B)
    # breakpoint()
    cr = Correlation(matrix_A_B, matrix_GT)

    return cr


def New_DC(latent_A, ground_truth):
    matrix_A = P_Distance_Matrix(latent_A)
    matrix_GT = P_Distance_Matrix(ground_truth)
    cr = Correlation(matrix_A, matrix_GT)

    return cr

class Loss_DC(nn.Module):
    def __init__(self, alpha = 0.1):
        super(Loss_DC, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.ImageNetLabelEmbedding = torch.nn.parameter.Parameter(torch.load('ImageNet_Class_Embedding.pt'), requires_grad = False)
        print("Loss balance alpha is: ", alpha)

    def CE(self, logit, target):
        return self.ce(logit, target)


    def forward(self, outputs, featuresX, featuresY, targets):
        cls_loss = self.CE(outputs, targets)

        imagenet_embedding = self.ImageNetLabelEmbedding[targets]
        dc_loss = P_DC(featuresX, featuresY, imagenet_embedding)

        loss = cls_loss - self.alpha * dc_loss
        return loss, cls_loss, dc_loss