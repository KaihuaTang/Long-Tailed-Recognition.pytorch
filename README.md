# A Strong Single-Stage Baseline for Long-Tailed Problems

[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)

This project provides a strong single-stage baseline for Long-Tailed Classification (under ImageNet-LT, Long-Tailed CIFAR-10/-100 datasets), Detection, and Instance Segmentation (under LVIS dataset). It is also a PyTorch implementation of the **NeurIPS 2020 paper** [Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect](https://arxiv.org/abs/2009.12991), which proposes a general solution to remove the bad momentum causal effect for a variety of Long-Tailed Recognition tasks. The codes are organized into three folders: 
1. The [classification folder](classification) supports long-tailed classification on ImageNet-LT, Long-Tailed CIFAR-10/CIFAR-100 datasets.
2. The [lvis_old folder (deprecated)](lvis_old) supports long-tailed object detection and instance segmentation on LVIS V0.5 dataset, which is built on top of mmdet V1.1.
3. The latest version of long-tailed detection and instance segmentation is under [lvis1.0 folder](lvis1.0). Since both LVIS V0.5 and mmdet V1.1 are no longer available on their homepages, we have to re-implement our method on [mmdet V2.4](https://github.com/open-mmlab/mmdetection) using [LVIS V1.0 annotations](https://www.lvisdataset.org/dataset). 

# Slides
If you want to present our work in your group meeting / introduce it to your friends / seek answers for some ambiguous parts in the paper, feel free to use our slides. It has two versions: [one-hour full version](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/NeurIPS%202020%20Presentation%20-%20Full%20(1hr).pptx) and [five-minute short version](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/NeurIPS%202020%20Presentation%20-%20Short%20(5min).pptx).

# Installation
The classification part allows the lower version of the following requirements. However, in detection and instance segmentation (mmdet V2.4), I tested some lower versions of python and pytorch, which are all failed. If you want to try other environments, please check the updates of [mmdetection](https://github.com/open-mmlab/mmdetection).

### Requirements:
- PyTorch >= 1.6.0
- Python >= 3.7.0
- CUDA >= 10.1
- torchvision >= 0.7.0
- gcc version >= 5.4.0 

### Step-by-step installation
```bash
conda create -n longtail pip python=3.7 -y
source activate longtail
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install pyyaml tqdm matplotlib sklearn h5py

# download the project
git clone https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch.git
cd Long-Tailed-Recognition.pytorch

# the following part is only used to build mmdetection 
cd lvis1.0
pip install mmcv-full
pip install mmlvis
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```
### Additional Notes
When we wrote the paper, we are using lvis V0.5 and mmdet V1.1 for our long-tailed instance segmentation experiments, but they've been deprecated by now. If you want to reproduce our results on lvis V0.5, you have to find a way to build mmdet V1.1 environments and use the code in lvis_old folder.


# Datasets
### ImageNet-LT
ImageNet-LT is a long-tailed subset of original ImageNet, you can download the dataset from its [homepage](http://image-net.org/index). After you download the dataset, you need to change the data_root of 'ImageNet' in [./classification/main.py](classification/main.py) file.

### CIFAR-10/-100
When you run the code for the first time, our dataloader will automatically download the CIFAR-10/-100. You need to set the data_root in [./classification/main.py](classification/main.py) to the path where you want to put all CIFAR data.

### LVIS
[Large Vocabulary Instance Segmentation (LVIS)](https://www.lvisdataset.org/) dataset uses the COCO 2017 train, validation, and test image sets. If you have already downloaded the COCO images, you only need to download the LVIS annotations. LVIS val set contains images from COCO 2017 train in addition to the COCO 2017 val split.

You need to put all the annotations and images under ./data/LVIS like this:
```bash
data
  |-- LVIS
    |--lvis_v1_train.json
    |--lvis_v1_val.json
      |--images
        |--train2017
          |--.... (images)
        |--test2017
          |--.... (images)
        |--val2017
          |--.... (images)
```

# Getting Started
For long-tailed classification, please go to [\[link\]](classification)

For long-tailed object detection and instance segmentation, please go to [\[link\]](lvis1.0)


# Advantages of the Proposed Method
- Compared with previous state-of-the-art [Decoupling](https://github.com/facebookresearch/classifier-balancing), our method only requires one-stage training.
- Most of the existing methods for long-tailed problems are using data distribution to conduct re-sampling or re-weighting during training, which is based on an inappropriate assumption that you can know the future distribution before you start to learn. Meanwhile, the proposed method doesn't need to know the data distribution during training, we only need to use an average feature for inference after we train the model.
- Our method can be easily transferred to any tasks. We outperform the previous state-of-the-arts [Decoupling](https://arxiv.org/abs/1910.09217), [BBN](https://arxiv.org/abs/1912.02413), [OLTR](https://arxiv.org/abs/1904.05160) in image classification, and we achieve better results than 2019 Winner of LVIS challenge [EQL](https://arxiv.org/abs/2003.05176) in long-tailed object detection and instance segmentation (under the same settings with even fewer GPUs).

# Citation
If you find our paper or this project helps your research, please kindly consider citing our paper in your publications.
```bash
@inproceedings{tang2020longtailed,
  title={Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect},
  author={Tang, Kaihua and Huang, Jianqiang and Zhang, Hanwang},
  booktitle= {NeurIPS},
  year={2020}
}
```
