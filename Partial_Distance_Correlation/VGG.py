import timm
import torch


def VGG19_BN():
    model = timm.create_model('vgg19_bn', pretrained=True)
    model.eval()
    return model