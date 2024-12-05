# PGD-pytorch
This repo is consisted with three parts, one is the original version of PGD attack, and another one is advanced version of PGD attack which only affect one model, and the last part is how PGD attack works on CircuitNet Generative Models. 

In the current stage, we have finished the implementation of PGD attack on Vanilla CNN models, and my modified the Vanilla PGD attack to adapt the model difference, in which affect one model while doesn't impact the other model. Later on, we will deploy the PGD attack on CircuitNet Generative Models. 


## Part 1:

**A pytorch implementation of "[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)"**

## Summary
This code is a pytorch implementation of **PGD attack**   
In this code, I used above methods to fool [Inception v3](https://arxiv.org/abs/1512.00567).   
'[Giant Panda](http://www.image-net.org/)' used for an example.   
You can add other pictures with a folder with the label name in the 'data/imagenet'.    

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


# Part 3: Data-Center-Processing---PGD-attack-on-CircuitNet

Dataset now is based on the https://github.com/circuitnet/CircuitNet. CircuitNet is an open-source dataset dedicated to machine learning (ML) applications in electronic design automation (EDA). It has collected more than 20K samples from versatile runs of commercial design tools based on open-source designs with various features for multiple ML for EDA applications. 

About setup you can follow the instruction in the CircuitNet github. Or you can switch to the 'dataset' branch.

## Setup Instructions
### 1. Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- [PyTorch](https://pytorch.org/)
- Other dependencies:
  - `torchvision`
  - `tqdm`

Install dependencies using:
```bash
pip install torch torchvision tqdm
```

### 2. Dataset and Model Preparation
Dataset: Ensure your dataset is formatted to work with the build_dataset function from the project.
- Trained Models:
	- model_target.pth: The target model to be attacked.
	- model_protected.pth: The protected model to remain unaffected.


### 3. Configuration File
Create a JSON configuration file (e.g., config.json) to specify parameters such as:
```
{
  "batch_size": 32,
  "cpu": false,
  "arg_file": null,
  "test_mode": true,
  "model_target_path": "./path_to_target_model.pth",
  "model_protected_path": "./path_to_protected_model.pth",
  "save_path": "./results",
  "eps": 0.03,
  "alpha": 0.01,
  "lmd": 1,
  "max_iters": 40
}
```

### 4. Running the Code
To run the PGD attack and evaluate the models:
```
python test.py --arg_file config.json --model_target_path ./path_to_target_model.pth --model_protected_path ./path_to_protected_model.pth
```

### Output after a small epoches of training
The script outputs the accuracy of both models on adversarial:
```
===> Evaluation Results:
Model Target Accuracy on Adversarial Examples: 45.12%
Model Protected Accuracy on Adversarial Examples: 78.35%
```

### Explanation of Key Parameters
- eps: Maximum allowed perturbation for adversarial examples.
- alpha: Step size for each PGD iteration.
- lmd: Weight of the penalty term to minimize the effect on model_protected.
- max_iters: Number of iterations for the PGD attack.


### Example Use Case
You can adjust eps, alpha, and lmd to control the trade-off between attacking model_target and preserving model_protected. For instance:

Increase lmd to prioritize the performance of model_protected.


### File Structure
pgd_attack.py: Encapsulates the PGD attack logic in a reusable class.
test.py: Evaluates models using the PGD attack and calculates differential impacts.
build_dataset.py, build_model.py, and other utility files: Handles dataset loading, model creation, and configurations.


### License
This project is licensed under the MIT License. See the LICENSE file for details.