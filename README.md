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
## Experiment workflow
### Example Usage:
Change the configuration in [utils/config.py](https://github.com/Yitong999/PGD-Attack-on-CircuitNet/blob/0c175552d2125bf3b33f93989d0d3c01e33e6d95/routability_ir_drop_prediction/utils/configs.py) to fit your file path and adjust the hyper-parameter before starting.

Test

Congestion 

	python test.py --task congestion_gpdl --pretrained PRETRAINED_WEIGHTS_PATH

DRC

	python test.py --task drc_routenet --pretrained PRETRAINED_WEIGHTS_PATH --save_path work_dir/drc_routenet/ --plot_roc 

IR Drop

	python test.py --task irdrop_mavi --pretrained PRETRAINED_WEIGHTS_PATH --save_path work_dir/irdrop_mavi/ --plot_roc

Train

Congestion

	python train.py --task congestion_gpdl --save_path work_dir/congestion_gpdl/

DRC

	python train.py --task drc_routenet --save_path work_dir/drc_routenet/

IR Drop

	python train.py --task irdrop_mavi --save_path work_dir/irdrop_mavi/

Attack

Congestion(Change attack algorithm in code)

	python PGD_attack.py --task congestion_gpdl --save_path SAVE_PATH --pretrained PRETRAINED_WEIGHTS_PATH

### Evaluation and expected results:
The experiments should reproduce the following key results:

*Transformer models show highest vulnerability (163.2% loss increase under PGD)
*CNN models show lowest vulnerability (51.2% loss increase under PGD)
*UNet models show intermediate vulnerability (80.3% loss increase under PGD)

Visual outputs include:

*Training convergence plots
*Adversarial perturbation visualizations
*Congestion prediction comparisons
