from torch.utils.data.dataset import Dataset
import torch
from torch import nn
from torch.nn import functional as F


class ImageTensorDataset(Dataset):

    def __init__(self, named_tensors):
        assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
        self.named_tensors = named_tensors

    def __getitem__(self, index):
        item = {name: tensor[index] for name, tensor in self.named_tensors.items()}

        if 'img' in item:
            item['img'] = item['img'].float() / 255.0

        return item

    def __len__(self):
        return list(self.named_tensors.values())[0].size(0)

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