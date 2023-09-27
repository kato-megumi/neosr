from pathlib import Path
from torch import nn
import torch
from torch.nn import functional as F
from .arch_util import net_opt

from neosr.utils.registry import ARCH_REGISTRY
from neosr.utils.options import parse_options


# initialize options parsing
root_path = Path(__file__).parents[2]
opt, args = parse_options(root_path, is_train=True)
# set scale factor in network parameters
upscale, training = net_opt()

def conv_layer(in_channels, out_channels, kernel_size):
    padding = int((kernel_size - 1) / 2)
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return F.relu(torch.cat((x, -x), 1))


@ARCH_REGISTRY.register()
class anime4k(nn.Module):
    def __init__(self, block_depth=7, stack_list=5, num_feat=12, act="crelu", last=False):
        super(anime4k, self).__init__()
        if act == "crelu":
            factor = 2
            self.act = CReLU()
        elif act == "prelu":
            factor = 1
            self.act = nn.PReLU(num_parameters=num_feat)
        if type(stack_list) == int:
            stack_list = list(range(-stack_list, 0))
        self.stack_list = stack_list
        self.ps = nn.PixelShuffle(2)
        
        self.conv_head = conv_layer(3, num_feat, kernel_size=3)
        self.conv_mid = nn.ModuleList(
            [
                conv_layer(num_feat * factor, num_feat, kernel_size=3)
                for _ in range(block_depth - 1)
            ]
        )
        if last:
            self.conv_tail = conv_layer(factor * num_feat * len(stack_list), 12, kernel_size=3)
        else:
            self.conv_tail = conv_layer(factor * num_feat * len(stack_list), 12, kernel_size=1)

    def forward(self, x):
        out = self.act(self.conv_head(x))
        depth_list = [out]
        for conv in self.conv_mid:
            out = self.act(conv(out))
            depth_list.append(out)
        out = self.conv_tail(torch.cat([depth_list[i] for i in self.stack_list], 1))
        out = self.ps(out) + F.interpolate(x, scale_factor=2, mode="bilinear")
        return torch.clamp(out, max=1.0, min=0.0)
