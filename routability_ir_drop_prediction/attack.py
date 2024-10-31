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
        
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

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


def train():
    wandb.init(project='circuitnet', mode='offline')

    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['save_path'] = os.path.abspath(arg_dict['save_path'])
    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'],  'arg.json'), 'wt') as f:
      json.dump(arg_dict, f, indent=4)

    print('save_path:', arg_dict['save_path'])
    
    
    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False 

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        print('Using GPU')
        model = model.cuda()
    
    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimzer
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    # Build lr scheduler
    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000
    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:        
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                input.requires_grad = True

                prediction = model(input)

                original_loss = loss(prediction, target)

                model.zero_grad()
                original_loss.backward(retain_graph=True)

                input_grad = input.grad.data

                epsilon = 0.01
                perturbed_input = fgsm_attack(input, epsilon, input_grad)
                    
                adv_prediction = model(perturbed_input)

                adv_loss = loss(adv_prediction, target)

                total_loss = original_loss + 0.5 * adv_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                wandb.log({
                    "original_loss": original_loss.item(),
                    "adv_loss": adv_loss.item(),
                    "total_loss": total_loss.item(),
                    "timestep": iter_num
                })
                
                iter_num += 1
                epoch_loss += total_loss.item()
                
                bar.update(1)
                
                if iter_num % print_freq == 0:
                    bar.close() 
                    print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(
                        iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq
                    ))
                    break
            
            bar.close()

        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict['save_path'])
            with torch.no_grad():
                adv_img = perturbed_input[0].detach().cpu().numpy()
                adv_img = np.transpose(adv_img, (1,2,0))
                adv_img = (adv_img - adv_img.min()) / (adv_img.max() - adv_img.min())
                plt.imsave(f'/scratch/yc7900/eg/code/circuit_learning/CircuitNet/routability_ir_drop_prediction/imgs/{iter_num}_adversarial_image.png', adv_img)
                pred_img = adv_prediction[0,0].detach().cpu().numpy()
                pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
                plt.imsave(f'/scratch/yc7900/eg/code/circuit_learning/CircuitNet/routability_ir_drop_prediction/imgs/{iter_num}_adv_prediction.png', pred_img, cmap='gray')
                input_img = input[0]  # shape: [3, 256, 256]
                input_img = input_img.cpu().numpy()
                input_img = np.transpose(input_img, (1,2,0))
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

                plt.imsave(f'/scratch/yc7900/eg/code/circuit_learning/CircuitNet/routability_ir_drop_prediction/imgs/{iter_num}_input_image.png', input_img)

                target_img = target[0,0].cpu().numpy()  # shape: [256, 256]
                target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())

                plt.imsave(f'/scratch/yc7900/eg/code/circuit_learning/CircuitNet/routability_ir_drop_prediction/imgs/{iter_num}_target_image.png', target_img, cmap='gray')
                pred_img = prediction[0,0].detach().cpu().numpy()  # shape: [256, 256]
                pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
                plt.imsave(f'/scratch/yc7900/eg/code/circuit_learning/CircuitNet/routability_ir_drop_prediction/imgs/{iter_num}_prediction_image.png', pred_img, cmap='gray')
            print('Images saved')
        epoch_loss = 0
    
    wandb.finish()



if __name__ == "__main__":
    train()
