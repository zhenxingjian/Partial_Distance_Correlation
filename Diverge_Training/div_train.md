## Prepare torchvision
We use a slightly modified version of torchvision. After installing torchvision using pip, do
```
cp efficientnet.py path_to_torchvision/models/
cp mobilenetv3.py path_to_torchvision/models/
```

## Training
```python
python imagenet_main.py --network=resnet152
```

## Test the accuracy under adversarial attack
```python
python test_adv_attack_imagenet.py --network=resnet152 
```
