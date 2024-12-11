# CircuitNet Under Attack: A Systematic Study of Adversarial Vulnerabilities in EDA Machine Learning Models
This repo contains the implementation of adversarial attacks (FGSM and PGD) on machine learning models trained on the CircuitNet dataset for EDA tasks. It includes code for training and evaluating three neural architectures (CNN, UNet, and Transformer) for congestion prediction, along with scripts for generating and analyzing adversarial examples.


## Part 1:

**A pytorch implementation of "[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)"**

**Inspired by "[CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (EDA)](https://github.com/circuitnet/CircuitNet.git)"**

## How to access
The artifact is available on [GitHub](https://github.com/Yitong999/PGD-Attack-on-CircuitNet). Users can clone the repository using: 

	git clone https://github.com/Yitong999/PGD-Attack-on-CircuitNet
	cd PGD-Attack-on-CircuitNet

## Requirements
* python==3.6   
* numpy==1.14.2   
* pytorch==1.0.1   

## Important results not in the code
- Capacity(size of network) plays an important role in adversarial training. (p.9-10)
	- For only natural examples training, it increases the robustness against one-step perturbations.
	- For PGD adversarial training, small capacity networks fails.
	- As capacity increases, the model can fit the adversairal examples increasingly well.
	- More capacity and strong adversaries decrease transferability. (Section B)
- FGSM adversaries don't increase robustness for large epsilon(=8). (p.9-10)
	- The network overfit to FGSM adversarial examples.
- Adversarial training with PGD shows good enough defense results.(p.12-13)

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
