# Tensorflow Implementation of On the Versatile Uses of Partial Distance Correlation in Deep Learning - Independent Features Help Robustness (Diverge Training) 

## Results
### Quantitative Results
The test accuracy (%) of a model $f_2$ on the adversarial examples generated using $f_1$ with the same architecture. "Baseline": train without constraint. "Ours": $f_2$ is independent to $f_1$. "Clean": test accuracy without adversarial examples.

| Dataset | Network | Method | Clean | FGM $\epsilon=0.03$ | PGD $\epsilon=0.03$ | FGM $\epsilon=0.05$ | PGD $\epsilon=0.05$ | FGM $\epsilon=0.10$ | PGD $\epsilon=0.10$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CIFAR10 | Resnet 18 | Baseline | 89.14 | 72.10 | 66.34 | 62.00 | 49.42 | 48.23 | 27.41 |
| CIFAR10 | Resnet 18 | Ours     | 87.61 | **74.76** | **72.85** | **65.56** | **59.33** | **50.24** | **36.11** |
| ImageNet |  Mobilenet-v3-small |  Baseline  | 47.16    | 29.64     | 30.00    | 23.52       | 24.81      | 13.90  | 17.15 |
| ImageNet |  Mobilenet-v3-small |  Ours  | 42.34     | **34.47**     | **36.98**    | **29.53**       | **33.77**   | **19.53** | **28.04** |
| ImageNet | Efficientnet-B0 |  Baseline | 57.85  | 26.72  | 28.22  | 18.96 | 19.45 | 12.04 | 11.17 |
| ImageNet | Efficientnet-B0 |  Ours  | 55.82   | **30.42**   | **35.99**  | **22.05** | **27.56** | **14.16** | **17.62** |
| ImageNet |  Resnet 34 |  Baseline  | 64.01     | 52.62     | 56.61    | 45.45       | 51.11       | 33.75 | 41.70 |
| ImageNet |  Resnet 34 |  Ours  | 63.77     | **53.19**     | **57.18**   | **46.50**   | **52.28**   | **35.00** | **43.35** |
| ImageNet | Resnet 152 |  Baseline  | 66.88  | 56.56     | 59.19    | 50.61       | 53.49     | 40.50 | 44.49 |
| ImageNet | Resnet 152 |  Ours  | 68.04  | **58.34**    | **61.33**    | **52.59**      | **56.05**    | **42.61** | **47.17** |

### Qualitative Results
![Diverge Training](../Diverge_Training/result/diverge_training.png)

## Training
Currently only ResNet models are added. 
Feel free to add more models by adapting forward function to return also the logits.
Training checkpoint will be saved at ./checkpoint
For ResNet18 diverge training on cifar10:
```python
python training.py --network=resnet18 --dataset=cifar10
```
For ResNet18 diverge training on ImageNet:
```python
python training.py --network=resnet18 --dataset=imagenet
```

## Test the accuracy under adversarial attack
```python
python test_adv_attack.py --network=resnet18 --dataset=cifar10
```

```python
python test_adv_attack.py --network=resnet18 --dataset=imagenet
```
