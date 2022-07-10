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
First download the pretrained network as the target for generating adversarial examples from here and put them in the checkpoint folder (Alternatively you can run the training with --num_nets=1 and use the saved model as the target model). 
```python
python test_adv_attack_imagenet.py --network=resnet152 
```
