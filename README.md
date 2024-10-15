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


# Part 3:
pending ...
# Data-Center-Processing---PGD-attack-on-CircuitNet
