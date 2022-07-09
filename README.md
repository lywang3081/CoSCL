# [CoSCL: Cooperation of Small Continual Learners is Stronger than a Big One (ECCV2022)]() 

------
This code is the official implementation of our paper.


## **Execution Details**

### Requirements

- Python 3
- GPU 1080Ti / Pytorch 1.3.1+cu9.2 / CUDA 9.2

### Download Dataset
- CUB-200-2011 : https://github.com/visipedia/tf_classification/wiki/CUB-200-Image-Classification
- Tiny-ImageNet : ```cd download``` ```source download_tinyimgnet.sh```

Please download the datasets and put them into ```./dat/CUB_200_2011``` and ```./dat/tiny-imagenet-200```, respectively.

### Execution command
The commands to run most of the experiments are included in **script_classification.sh**.

Below, we present the demo commands for our method with EWC as the default continual learner. 

For small-scale images:

```
# CIFAR-100-SC
$ python3 ./main.py --experiment split_cifar100_sc_5 --approach ewc_coscl --lamb 40000 --lamb1 0.02 --use_TG --s_gate 100 --seed 0

# CIFAR-100-RS
$ python3 ./main.py --experiment split_cifar100_rs_5 --approach ewc_coscl --lamb 10000 --lamb1 0.02 --use_TG --s_gate 100 --seed 0

```

For large-scale images:

```
$ cd LargeScale_Image

# CUB-200-2011
$ python3 ./main.py --dataset CUB200 --trainer ewc_coscl --lamb 1 --lamb1 0.0001 --s_gate 100 --tasknum 10 --seed 0

# Tiny-ImageNet
$ python3 ./main.py --dataset tinyImageNet --trainer ewc_coscl --lamb 320 --lamb1 0.001 --s_gate 100 --tasknum 10 --seed 0

```

