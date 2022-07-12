import timm
import torch

def densenet121():
    model = timm.create_model('densenet121', pretrained=True)
    model.eval()
    return model