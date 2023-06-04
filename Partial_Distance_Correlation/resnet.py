import timm
import torch

def resnet18():
    model = timm.create_model('resnet18', pretrained=True)
    model.eval()
    return model

def resnet34():
    model = timm.create_model('resnet34', pretrained=True)
    model.eval()
    return model
    

def resnet50():
    model = timm.create_model('resnet50', pretrained=True)
    model.eval()
    return model


def resnet152():
    model = timm.create_model('resnet152', pretrained=True)
    model.eval()
    return model


    
def resnext50():
    model = timm.create_model('resnext50_32x4d', pretrained=True)
    model.eval()
    return model