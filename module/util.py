import torch.nn as nn
from module.mlp import MLP
from module.CNN import CNN

def get_model(model_tag, num_classes, device):
    if model_tag == "MLP":
        return MLP(num_classes=num_classes, device=device)
    elif model_tag == "CNN":
        return CNN(num_classes=num_classes, device=device)
    else:
        raise NotImplementedError