# On the Versatile Uses of Partial Distance Correlation in Deep Learning - Disentanglement implementation

Note: this repo is heavily based on the baseline paper:
> [An Image is Worth More Than a Thousand Words: Towards Disentanglement in the Wild](http://www.vision.huji.ac.il/zerodim)  
> Aviv Gabbay, Niv Cohen and Yedid Hoshen  

## Human Face Manipulation (FFHQ) Results

### Visulization

### Quantitive measurement (distance correlation between residual and attribute of interest)

DC between residual attributes (R) and attributes of interest, if we use the ground truth CLIP labeled data to measure the attribute of interest.

| age vs R. | gender vs R. | ethnicity vs R. | hair color vs R. | beard vs R. | glasses vs R |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0329 | 0.0180 | 0.0222 | 0.0242 | 0.0219 | 0.0255 |

DC between residual attributes (R) and attributes of interest, if we use in-model classifier to classify the attribute of interest.

| age vs R. | gender vs R. | ethnicity vs R. | hair color vs R. | beard vs R. | glasses vs R |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0430 | 0.0124 | 0.0376 | 0.0259 | 0.0490 | 0.0188 |
 


## Requirements
![python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
![pytorch 1.3](https://img.shields.io/badge/pytorch-1.3-orange.svg)
![cuda 10.1](https://img.shields.io/badge/cuda-10.1-green.svg)

This repository imports modules from the StyleGAN2 architecture (**not pretrained**).
Clone the following repository:
```
git clone https://github.com/rosinality/stylegan2-pytorch
```
Add the local StyleGAN2 project to PYTHONPATH. For bash users:
```
export PYTHONPATH=<path-to-stylegan2-project>
```

## Training
In order to train a model from scratch, do the following preprocessing and training steps.
First, create a directory (can be specified by `--base-dir` or set to current working directory by default) for the training artifacts (preprocessed data, models, training logs, etc).

### Zero-shot Disentangled Image Manipulation
For training a model on real images, without any manual annotations, first prepare an unlabeled image dataset. E.g. for FFHQ, download the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset), create a local directory named `ffhq-dataset` with all the png images placed in a single `imgs` subdir, and apply the following preprocessing:
```
python main.py preprocess --dataset-id ffhq --dataset-path ffhq-dataset --out-data-name ffhq-x256
```

The images are then annotated in a zero-shot manner using [CLIP](https://github.com/openai/CLIP). Clone the external repository and add to your local path:
```
git clone https://github.com/openai/CLIP
export PYTHONPATH=<path-to-clip-project>
```

For example, human face images can be annotated as follows:
```
python annotate_with_clip.py --data-name ffhq-x256 --K 1000
```

The model can then be trained as follows:
```
python main.py train --config ffhq_clip --data-name ffhq-x256 --model-name DC_Disentanglement
```

#### Resources
The training automatically detects all the available gpus and applies multi-gpu mode if available.

#### Logs
During training, loss metrics and translation visualizations are logged with tensorboard and can be viewed by:
```
tensorboard --logdir <base-dir>/cache/tensorboard --load_fast true
```

## Pretrained Models
We provide a model pretrained on human face images ([FFHQ](https://github.com/NVlabs/ffhq-dataset)), with partial annotations obtained in a zero-shot manner using [CLIP](https://github.com/openai/CLIP) for the following attributes: `age, gender, ethnicity, hair_color, beard, glasses`. Download the entire model directory from [distance_correlation](https://drive.google.com/file/d/1AWmR04aQqi9OZJcrxfIajuJZJtSiUm1I/view?usp=sharing) and place it under `<base-dir>/cache/models`.

## Inference
Given a trained model (either pretrained or trained from scratch), a test image named as *input.png*, can be manipulated as follows:
```
bash evaluate.sh
```

**Note:** Face manipulation models are very sensitive to the face alignment. The target face should be aligned exactly as done in the pipeline which CelebA-HQ and FFHQ were created by. Use the alignment method implemented [here](https://github.com/Puzer/stylegan-encoder/blob/master/align_images.py) before applying any of the human face manipulation models on external images.

## Inference DC
Given a trained model (either pretrained or trained from scratch), we can measure the distance correlation between residual and attributes of interest
```
CUDA_VISIBLE_DEVICES=0 python correlation_BERT.py evaluate --config ffhq_clip --data-name ffhq-x256 --model-name DC_Disentanglement 
```
