import numpy as np
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.transforms as transforms


from scipy.stats import gaussian_kde


plt.style.use('ggplot')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'


def Distance_Correlation(latent, control):

    # latent = F.normalize(latent)
    # control = F.normalize(control)

    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)


    matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r


def Peasor_Correlation(latent, control):
    batch_size = latent.shape[0]

    up = (latent - torch.mean(latent, dim = 0, keepdims= True) ) * (control - torch.mean(control, dim = 0, keepdims= True) )
    up = torch.sum(up) / batch_size


    down = torch.sum((latent - torch.mean(latent, dim = 0, keepdims= True) ) ** 2 ) * torch.sum((control - torch.mean(control, dim = 0, keepdims= True) ) ** 2)
    down = down / (batch_size ** 2)

    return up/torch.sqrt(down)


def confidence_ellipse(x, y, ax, n_std=2.9, facecolor='none', cov=None, **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, lw=3, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    batch_size=10000
    x = np.linspace(-3, 3, num=batch_size)
    y = np.random.randn(batch_size)

    y = y + x**2

    x = torch.Tensor(x)
    x = x.reshape([batch_size,-1])
    y = torch.Tensor(y)
    y = y.reshape([batch_size,-1])

    pc = Peasor_Correlation(x,y)

    print(pc)

    dc = Distance_Correlation(x,y)

    print(dc)

    outR = torch.cat([x,y], axis=-1).numpy()

    z = gaussian_kde(outR.transpose())(outR.transpose())
    figure(figsize=(3, 4.5), dpi=80)
    new_X = np.linspace(-3, 3, num=batch_size)
    new_Y_low = new_X**2 - 3
    new_Y_high = new_X**2 + 3
    plt.plot(new_X, new_X**2, 'b-', alpha=0.8)
    ax = plt.gca()
    ax.fill_between(new_X, new_Y_low, new_Y_high, color='b' , alpha=0.1)
    plt.scatter(x=x[::20], y = y[::20],c=z[::20],s=10,cmap='hot')
    ax.axis('equal')
    ax.set(xlim=(-5, 5), ylim=(-5, 15))
    # ax.set_title(r'$y=x^2+n,n\sim \mathcal{N}(0,1)$'))
    plt.text(0.05, 0.95, 'Pearson Cor : {:0.3f}\nDistance Cor: {:0.3f}'.format(pc, dc) , horizontalalignment='left',verticalalignment='center', transform=ax.transAxes, size='x-large')
    plt.savefig('pearson_distance1.png', dpi=300)
    plt.show()



    xy = np.random.multivariate_normal(mean=np.asarray([0,5]), cov = np.asarray([[1,1.5],[1.5,5]]), size=batch_size)
    x = xy[:,0]
    y = xy[:,1]

    x = torch.Tensor(x)
    x = x.reshape([batch_size,-1])
    y = torch.Tensor(y)
    y = y.reshape([batch_size,-1])

    pc = Peasor_Correlation(x,y)

    print(pc)

    dc = Distance_Correlation(x,y)

    print(dc)

    outR = torch.cat([x,y], axis=-1).numpy()
    z = gaussian_kde(outR.transpose())(outR.transpose())

    figure(figsize=(3, 4.5), dpi=80)
    plt.scatter(x=x[::10], y = y[::10],c=z[::10],s=10,cmap='hot')
    ax = plt.gca()
    confidence_ellipse(outR[:,0], outR[:,1],ax, cov=np.asarray([[1,1.5],[1.5,5]]), edgecolor=None, facecolor='blue', alpha=0.1)
    ax.axis('equal')
    ax.set(xlim=(-5, 5), ylim=(-5, 15))
    plt.text(0.05, 0.95, 'Pearson Cor : {:0.3f}\nDistance Cor: {:0.3f}'.format(pc, dc) , horizontalalignment='left',verticalalignment='center', transform=ax.transAxes, size='x-large')
    plt.savefig('pearson_distance2.png', dpi=300)
    plt.show()


    xy = np.random.multivariate_normal(mean=np.asarray([0,5]), cov = np.asarray([[1,0],[0,5]]), size=batch_size)
    x = xy[:,0]
    y = xy[:,1]

    x = torch.Tensor(x)
    x = x.reshape([batch_size,-1])
    y = torch.Tensor(y)
    y = y.reshape([batch_size,-1])

    pc = Peasor_Correlation(x,y)

    print(pc)

    dc = Distance_Correlation(x,y)

    print(dc)

    outR = torch.cat([x,y], axis=-1).numpy()
    z = gaussian_kde(outR.transpose())(outR.transpose())

    figure(figsize=(3, 4.5), dpi=80)
    plt.scatter(x=x[::10], y = y[::10],c=z[::10],s=10,cmap='hot')
    ax = plt.gca()
    confidence_ellipse(outR[:,0], outR[:,1],ax, cov=np.asarray([[1,0],[0,5]]), edgecolor=None, facecolor='blue', alpha=0.1)
    ax.axis('equal')
    ax.set(xlim=(-5, 5), ylim=(-5, 15))
    plt.text(0.05, 0.95, 'Pearson Cor : {:0.3f}\nDistance Cor: {:0.3f}'.format(pc.abs(), dc) , horizontalalignment='left',verticalalignment='center', transform=ax.transAxes, size='x-large')
    plt.savefig('pearson_distance3.png', dpi=300)
    plt.show()

