# Copyright 2022 CircuitNet. All rights reserved.

import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from models.build_model import build_model
from utils.configs import Parser
from math import cos, pi
import sys, os, subprocess
import wandb

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
        


class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')


    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    
    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(image, epsilon, model, target, loss_fn, alpha=0.01, num_steps=10):
    perturbed_image = image.clone().detach()
    
    for i in range(num_steps):
        perturbed_image.requires_grad = True
        prediction = model(perturbed_image)
        
        # Handle multiple losses
        if isinstance(loss_fn(prediction, target), tuple):
            pixel_loss, _ = loss_fn(prediction, target)
        else:
            pixel_loss = loss_fn(prediction, target)
        
        pixel_loss.backward()
        
        with torch.no_grad():
            grad_sign = perturbed_image.grad.sign()
            perturbed_image = perturbed_image + alpha * grad_sign

            delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1)
        
        perturbed_image = perturbed_image.detach()
        
    return perturbed_image


def train():
    wandb.init(project='circuitnet', mode='offline')

    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    # Setup save path
    arg_dict['save_path'] = os.path.abspath(arg_dict['save_path'])
    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'], 'arg.json'), 'wt') as f:
        json.dump(arg_dict, f, indent=4)

    print('save_path:', arg_dict['save_path'])
    
    # Setup dataset
    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False 
    print('===> Loading datasets')
    arg_dict['num_workers'] = min(8, arg_dict.get('num_workers', 8))
    dataset = build_dataset(arg_dict)

    # Build model
    print('===> Building model')
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        print('Using GPU')
        model = model.cuda()
    
    # Setup loss, optimizer and scheduler
    loss = build_loss(arg_dict)
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'], betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])
    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    # Training parameters
    iter_num = 0
    epoch_loss = 0
    print_freq = 100
    save_freq = 10000
    
    # Adversarial attack parameters
    epsilon = 0.01  # FGSM/PGD perturbation size
    alpha = 0.01    # PGD step size
    num_steps = 10  # PGD number of steps
    attack_type = "fgsm"  #  pgd or "fgsm"

    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:        
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()
                
                # Update learning rate
                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                # Clean prediction and loss
                prediction = model(input)
                
                if isinstance(arg_dict['loss_type'], list):
                    clean_pixel_loss, clean_separate_losses = loss(prediction, target)
                    clean_mse_loss = clean_separate_losses[0]
                    clean_ssim_loss = clean_separate_losses[1]
                else:
                    clean_pixel_loss = loss(prediction, target)
                    clean_mse_loss = clean_pixel_loss

                # Generate adversarial examples using PGD
                perturbed_input = pgd_attack(
                    input, 
                    epsilon, 
                    model, 
                    target, 
                    loss,
                    alpha=alpha,
                    num_steps=num_steps
                )

                # Adversarial prediction and loss
                adv_prediction = model(perturbed_input)
                
                if isinstance(arg_dict['loss_type'], list):
                    adv_pixel_loss, adv_separate_losses = loss(adv_prediction, target)
                    adv_mse_loss = adv_separate_losses[0]
                    adv_ssim_loss = adv_separate_losses[1]
                else:
                    adv_pixel_loss = loss(adv_prediction, target)
                    adv_mse_loss = adv_pixel_loss

                # Log losses
                wandb.log({
                    "clean_mse_loss": clean_mse_loss.item(),
                    "clean_ssim_loss": clean_ssim_loss.item() if isinstance(arg_dict['loss_type'], list) else 0,
                    "clean_total_loss": clean_pixel_loss.item(),
                    "adv_mse_loss": adv_mse_loss.item(),
                    "adv_ssim_loss": adv_ssim_loss.item() if isinstance(arg_dict['loss_type'], list) else 0,
                    "adv_total_loss": adv_pixel_loss.item(),
                    "lr": regular_lr,
                    "timestep": iter_num
                })
                
                epoch_loss += clean_mse_loss.item()

                # Optimization step
                optimizer.zero_grad()
                total_loss = clean_pixel_loss + 0.5 * adv_pixel_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                iter_num += 1
                bar.update(1)

                if iter_num % print_freq == 0:
                    break

        print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(
            iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
        
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict['save_path'])
            
        epoch_loss = 0
    
    wandb.finish()

if __name__ == "__main__":
    train()
