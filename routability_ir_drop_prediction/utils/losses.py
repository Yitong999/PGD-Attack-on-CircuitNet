# Copyright 2022 CircuitNet. All rights reserved.

import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.losses as losses


# def build_loss(opt):
#     return losses.__dict__[opt.pop('loss_type')]()

# def build_loss(opt):
#     """Build loss function from options"""
#     opt = opt.copy()  
#     loss_types = opt.pop('loss_type')
#     loss_weights = opt.pop('loss_weights', None)
    
#     if isinstance(loss_types, str):
#         return losses.__dict__[loss_types]()

#     elif isinstance(loss_types, list):
#         if loss_weights is None:
#             loss_weights = [1.0] * len(loss_types)
            
#         loss_functions = []
#         for loss_type in loss_types:
#             loss_functions.append(losses.__dict__[loss_type]())
            
#         def combined_loss(pred, target):
#             total_loss = 0
#             for weight, loss_fn in zip(loss_weights, loss_functions):
#                 total_loss += weight * loss_fn(pred, target)
#             return total_loss
            
#         return combined_loss
    
#     else:
#         raise TypeError(f"Unsupported loss_type: {type(loss_types)}")

def build_loss(opt):
    """Build loss function from options"""
    opt = opt.copy()
    loss_types = opt.pop('loss_type')
    loss_weights = opt.pop('loss_weights', None)

    if isinstance(loss_types, str):
        return losses.__dict__[loss_types]()

    elif isinstance(loss_types, list):
        if loss_weights is None:
            loss_weights = [1.0] * len(loss_types)
            
        loss_functions = []
        for loss_type in loss_types:
            loss_functions.append(losses.__dict__[loss_type]())
            
        def combined_loss(pred, target):
            separate_losses = []
            weighted_losses = []
            for weight, loss_fn in zip(loss_weights, loss_functions):
                curr_loss = loss_fn(pred, target)
                separate_losses.append(curr_loss)
                weighted_losses.append(weight * curr_loss)

            return sum(weighted_losses), separate_losses
            
        return combined_loss
    else:
        raise TypeError(f"Unsupported loss_type: {type(loss_types)}")

__all__ = ['L1Loss', 'MSELoss', 'SSIMLoss']


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()

    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12

        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss

def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                sample_wise=False,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper

@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@masked_loss
def ssim_loss(pred, target):
    """Wrapper for SSIM loss to match the interface of other losses"""
    ssim = SSIMLoss()
    return ssim(pred, target)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                            for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        img1 = torch.clamp(img1, min=0, max=1)
        img2 = torch.clamp(img2, min=0, max=1)
        
        (_, channel, _, _) = img1.size()
        window = self.window
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            loss = 1 - ssim_map.mean()
        else:
            loss = 1 - ssim_map
            
        return loss


class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)



class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)