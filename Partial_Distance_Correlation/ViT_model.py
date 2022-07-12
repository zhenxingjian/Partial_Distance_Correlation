import timm
import torch

def ViT():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    return model


def ViT_large():
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    model.eval()
    return model


def ViT_hy():
    model = timm.create_model('vit_large_r50_s32_224', pretrained=True)
    model.eval()
    return model

    


def DeiT():
    model = timm.create_model('deit_base_patch16_224', pretrained=True)
    model.eval()
    return model