# Copyright 2022 CircuitNet. All rights reserved.

import models
import torch

def build_model(opt):
    model_type = opt.pop('model_type')
    
    if model_type == 'UNet':
        model_params = {
            'in_channels': opt.pop('in_channels'),
            'out_channels': opt.pop('out_channels'),
            'num_filters': opt.pop('num_filters', 32) 
        }
        model = models.__dict__[model_type](**model_params)
    elif model_type == 'CircuitTransformer':
        model_params = {
            'image_size': opt.pop('image_size', 256),
            'patch_size': opt.pop('patch_size', 16),
            'in_channels': opt.pop('in_channels'),
            'out_channels': opt.pop('out_channels'),
            'embed_dim': opt.pop('embed_dim', 512),
            'depth': opt.pop('depth', 12),
            'heads': opt.pop('heads', 8),
            'dim_head': opt.pop('dim_head', 64),
            'mlp_ratio': opt.pop('mlp_ratio', 4.),
            'dropout': opt.pop('dropout', 0.1)
        }
        model = models.__dict__[model_type](**model_params)
    else:
        model = models.__dict__[model_type](**opt)

    model.init_weights(**opt)
    if opt.get('test_mode', False): 
        model.eval()
        
    return model

# def build_model(opt):
#     model_type = opt.pop('model_type')
    
#     if model_type == 'UNet':
#         model_params = {
#             'in_channels': opt.pop('in_channels'),
#             'out_channels': opt.pop('out_channels'),
#             'num_filters': opt.pop('num_filters', 32) 
#         }
#         model = models.__dict__[model_type](**model_params)
#     else:
#         model = models.__dict__[model_type](**opt)

#     model.init_weights(**opt)
#     if opt.get('test_mode', False): 
#         model.eval()
        
#     return model

# def build_model(opt):
#     model = models.__dict__[opt.pop('model_type')](**opt)
#     model.init_weights(**opt)
#     if opt['test_mode']:
#         model.eval()
#     return model
