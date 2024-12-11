# CircuitNet Under Attack: A Systematic Study of Adversarial Vulnerabilities in EDA Machine Learning Models
This repo contains the implementation of adversarial attacks (FGSM and PGD) on machine learning models trained on the CircuitNet dataset for EDA tasks. It includes code for training and evaluating three neural architectures (CNN, UNet, and Transformer) for congestion prediction, along with scripts for generating and analyzing adversarial examples.


## Part 1:

**A pytorch implementation of "[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)"**

**Inspired by "[CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (EDA)](https://github.com/circuitnet/CircuitNet.git)"**

## How to access
The artifact is available on [GitHub](https://github.com/Yitong999/PGD-Attack-on-CircuitNet). Users can clone the repository using: 

	git clone https://github.com/Yitong999/PGD-Attack-on-CircuitNet
	cd PGD-Attack-on-CircuitNet

## Hardware dependencies
* NVIDIA GPU with 12GB+ VRAM (tested on NVIDIA A100)
* 32GB+ RAM
* 100GB+ free disk space 

## Software dependencies
* python==3.8+   
* numpy==1.14.2   
* pytorch==1.11

## Prerequisites
Dependencies can be installed using pip:

	pip install -r requirements.txt

PyTorch is not included in requirement.txt, and you could install it following the instruction on PyTorch homepage [https://pytorch.org/](https://pytorch.org/).

DGL is also not included in requirement.txt, and it is required for net delay prediction only. You could install it following the instruction on DGL homepage [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html).

Our experiments run on Python 3.9 and PyTorch 1.11. Other versions should work but are not tested.

## Data sets
Please follow the instructions on the [download page](https://circuitnet.github.io/intro/download.html) to set up the CircuitNet dataset for a specific task(Congestion/DRC/IR Drop).

Dataset download links:

[Baidu Netdisk.](https://pan.baidu.com/share/init?surl=udXVZnfjqniH9paKfyc2eQ&pwd=ijdh)

[Google Drive](https://drive.google.com/drive/folders/10PD4zNa9fiVeBDQ0-drBwZ3TDEjQ3gmf)

## Models
Implementation includes three neural architectures:

* CNN (baseline)
* UNet
* Transformer

## Notice
- This Repository won't be updated.
- Please check [the package of adversarial attacks in pytorch](https://github.com/Harry24k/adversairal-attacks-pytorch)



# Part 2:
### Download traffic sign GERMAN from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data, rename it as German

and run below script to keep only 10 classes for a easier training
```
python preprocess.py
```
### Train models:
train and save MLP model:
```
python train.py --epochs=50 --model=MLP --save_name=MLP_50
```

train and save CNN model:
```
python train.py --epochs=20 --model=CNN --save_name=MLP_20
```

### Advanced PGD attack:
```
python eval_after_pgd.py
```


# Part 3:
pending ...
# Data-Center-Processing---PGD-attack-on-CircuitNet
