import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .utils import load_state_dict, generation_init_weights


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn
    )
    return model

def up_conv(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_filters=32):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        
        # Encoder
        self.down_1 = conv_block_2(self.in_channels, self.num_filters, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filters, self.num_filters*2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filters*2, self.num_filters*4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filters*4, self.num_filters*8, act_fn)
        self.pool_4 = maxpool()
        
        # Bridge
        self.bridge = conv_block_2(self.num_filters*8, self.num_filters*16, act_fn)
        
        # Decoder
        self.up_1 = up_conv(self.num_filters*16, self.num_filters*8, act_fn)
        self.up_conv_1 = conv_block_2(self.num_filters*16, self.num_filters*8, act_fn)
        self.up_2 = up_conv(self.num_filters*8, self.num_filters*4, act_fn)
        self.up_conv_2 = conv_block_2(self.num_filters*8, self.num_filters*4, act_fn)
        self.up_3 = up_conv(self.num_filters*4, self.num_filters*2, act_fn)
        self.up_conv_3 = conv_block_2(self.num_filters*4, self.num_filters*2, act_fn)
        self.up_4 = up_conv(self.num_filters*2, self.num_filters, act_fn)
        self.up_conv_4 = conv_block_2(self.num_filters*2, self.num_filters, act_fn)
        
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(self.num_filters, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)
        
        # Bridge
        bridge = self.bridge(pool_4)
        
        # Decoding
        up_1 = self.up_1(bridge)
        up_conv_1 = self.up_conv_1(torch.cat([up_1, down_4], dim=1))
        up_2 = self.up_2(up_conv_1)
        up_conv_2 = self.up_conv_2(torch.cat([up_2, down_3], dim=1))
        up_3 = self.up_3(up_conv_2)
        up_conv_3 = self.up_conv_3(torch.cat([up_3, down_2], dim=1))
        up_4 = self.up_4(up_conv_3)
        up_conv_4 = self.up_conv_4(torch.cat([up_4, down_1], dim=1))
        
        # Output
        out = self.out(up_conv_4)
        return out

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        """Initialize weights for the model"""
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         nn.init.normal_(m.weight, 0.0, 0.02)
            #         if m.bias is not None:
            #             nn.init.constant_(m.bias, 0)
            #     elif isinstance(m, nn.BatchNorm2d):
            #         nn.init.normal_(m.weight, 1.0, 0.02)
            #         nn.init.constant_(m.bias, 0)
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None.")