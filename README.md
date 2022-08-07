# On the Versatile Uses of Partial Distance Correlation in Deep Learning - Official PyTorch Implementation
> [On the Versatile Uses of Partial Distance Correlation in Deep Learning](https://arxiv.org/abs/2207.09684)  
> Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh 
> European Conference on Computer Vision (ECCV), 2022.

> **Abstract:** Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. 

<a href="https://arxiv.org/abs/2207.09684" target="_blank"><img src="https://img.shields.io/badge/arXiv-2207.09684-b31b1b.svg"></a>


## Results

### Independent Features Help Robustness (Diverge Training)

### Informative Comparisons between Networks (Partial Distance Correlation)

Remove model Y from model X, and compute the correlation between the residual and the ground truth label embedding.


| Network $\Theta_X$ |  Network $\Theta_Y$ | $\mathcal{R}^2(X, GT)$ | $\mathcal{R}^2(Y, GT)$ | $\mathcal{R}^2(X\|Y, GT)$ | $\mathcal{R}^2((Y\|X), GT)$
|:---:|:---:|:---:|:---:|:---:|:---:|
| ViT$^1$     |  Resnet 18$^2$   |  0.042     |  0.025    |  0.035       |  0.007 |
| ViT         |  Resnet 50$^3$   |  0.043     |  0.036    |  0.028       |  0.017 |
| ViT         |  Resnet 152$^4$  |  0.044     |  0.020    |  0.040       |  0.009 |
| ViT         |  VGG 19 BN$^5$  |  0.042     |  0.037    |  0.026       |  0.015 |
| ViT         |  Densenet121$^6$ |  0.043     |  0.026    |  0.035       |  0.007 |
| ViT large$^7$   |  Resnet 18   |  0.046     |  0.027    |  0.038       |  0.007 |
| ViT large   |  Resnet 50   |  0.046     |  0.037    |  0.031       |  0.016 |
| ViT large   |  Resnet 152  |  0.046     |  0.021    |  0.042       |  0.010 |
| ViT large   |  ViT         |  0.045     |  0.043    |  0.019       |  0.013 |
| ViT+Resnet 50$^8$ |  Resnet 18  |  0.044     |  0.024    |  0.037       |  0.005 |
| Resnet 152  |  Resnet 18   |  0.019     |  0.025    |  0.013       |  0.020 |
| Resnet 152  |  Resnet 50   |  0.021     |  0.037    |  0.003       |  0.030 |
| Resnet 50   |  Resnet 18   |  0.036     |  0.025    |  0.027       |  0.008 |
| Resnet 50   |  VGG 19 BN   |  0.036     |  0.036    |  0.020       |  0.019| 

*Note Accuracy: 1. 84.40%; 2. 69.76%; 3. 79.02%; 4. 82.54%; 5. 74.22%; 6. 75.57%; 7. 85.68%; 8. 84.13%*



### Disentanglement

*Visualization*
![Distance Correlation in Disentanglement](Disentanglement/result/Disentanglement_result.png)

*Quantitive measurement (distance correlation between residual and attribute of interest)*

DC between residual attributes (R) and attributes of interest, if we use the ground truth CLIP labeled data to measure the attribute of interest.

| age vs R. | gender vs R. | ethnicity vs R. | hair color vs R. | beard vs R. | glasses vs R |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0329 | 0.0180 | 0.0222 | 0.0242 | 0.0219 | 0.0255 |

DC between residual attributes (R) and attributes of interest, if we use in-model classifier to classify the attribute of interest.

| age vs R. | gender vs R. | ethnicity vs R. | hair color vs R. | beard vs R. | glasses vs R |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0430 | 0.0124 | 0.0376 | 0.0259 | 0.0490 | 0.0188 |
