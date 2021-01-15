# Image compression

This project reimplements the work of Balle et al. (2018), [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436), which is an end-to-end learning-based image compression model that incorporates a scale hyperprior to effeciently encode spatial dependencies in the encoded representations. This code is based on the official [Tensorflow implementation](https://github.com/tensorflow/compression)

## Results
The model was trained on 30000 randomly chosen images from ImageNet (uniformly from 1000 classes) and evaluated on [Kodak dataset](http://r0k.us/graphics/kodak/)
### Peak signal-to-noise ratio
![](https://github.com/hieu1999210/image_compression/blob/master/figures/psnr.png)

### Multi-scale structural similarity index measure
MS-SSIM was converted to decibels by the following formular -10log(1 - MS-SSIM)

![](https://github.com/hieu1999210/image_compression/blob/master/figures/ms_ssim.png)


## Installation

### Requirements
- torch >= 1.6
- Pillow = 7.2
- tqdm
- tensorboard

### Usage

#### Data preparation
- ImageNet

images folder structure
```bash
    path-to-imagenet-folder/
    ├── accompanist
    │   ├── 150795375_5519d9ee7a.jpg
    │   ├── 1580967932_2e4f82b013.jpg
    │   ├── 175603732_ccaff839b3.jpg
    │   ├── ...
    |   ├── 69754088_a93ab2d88c.jpg
    |   ├── 763161837_ba14aa3907.jpg
    |   └── 809481545_6086b1fcd3.jpg
    ├── ...
    └── zooplankton
        ├── 1166318885_afb3268a6e.jpg
        ├── 1166319437_104650fdd3.jpg
        ├── 1167178280_a680f0abb5.jpg
        ├── ...
        ├── 367485857_0f19d03246.jpg
        ├── 367485874_4c183f7684.jpg
        └── 407992561_1b2711892f.jpg
```

metadata.csv
```bash
path
accompanist/150795375_5519d9ee7a.jpg
accompanist/1580967932_2e4f82b013.jpg
accompanist/175603732_ccaff839b3.jpg
...
zooplankton/367485857_0f19d03246.jpg
zooplankton/367485874_4c183f7684.jpg
zooplankton/407992561_1b2711892f.jpg
```

- Kodak
```bash
    path-to-Kodak-folder/
    ├── 1.png
    ├── 2.png
    ├── 3.png
    ...
    ├── 22.png
    ├── 23.png
    └── 24.png
```

For training or evaluation DIRS in config files need to be changed to the correct path.

#### Training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config './configs/<<config_file>>' \
    
```
For resuming training, add flag --resume

#### Tensorboard
```bash
tensorboard --logdir=<<path-to-output-folder>>
```
#### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config './configs/<<config_file>>' \
    --load '<<path-to-best-cp>>' \
    --valid \
    --save_output # optional
```

# Acknowledgment
The ImageNet dataset was crawled using [ImageNet-datasets-downloader](https://github.com/mf1024/ImageNet-datasets-downloader)
