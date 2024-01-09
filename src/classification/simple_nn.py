import torch
import torch.nn as nn
import torch.nn.functional as F

from rtdl_revisiting_models import ResNet


# https://github.com/yandex-research/rtdl-revisiting-models/tree/main
class ResNetNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.ModuleList([
            ResNet(d_in=input_size, d_out=None, n_blocks=1, d_block=32, d_hidden=64, d_hidden_multiplier=None, dropout1=0.15, dropout2=0.0,),
            ResNet(d_in=32, d_out=None, n_blocks=1, d_block=64, d_hidden=128, d_hidden_multiplier=None, dropout1=0.15, dropout2=0.0,),
            ResNet(d_in=64, d_out=output_size, n_blocks=1, d_block=128, d_hidden=256, d_hidden_multiplier=None, dropout1=0.15, dropout2=0.0,)
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        
        return x




class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size, last_layer=False):
        super().__init__()
        # batchnorm leads to bad results
        self.linear1 = nn.Linear(in_size, out_size)
        self.linear2 = nn.Linear(out_size, out_size)
        self.short_cut = nn.Linear(in_size, out_size)

        self.relu = nn.ReLU()

        self.last_layer = last_layer

    def forward(self, x):
        # residual = x
        residual = self.short_cut(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual 
        if not self.last_layer:
            out = self.relu(out)

        return out



class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, r1=256, r2=128, r3=64, r4=32):
        super().__init__()
        self.model = nn.ModuleList([
            ResidualBlock(input_size, r1),
            
            ResidualBlock(r1, r2),

            ResidualBlock(r2, r3),

            ResidualBlock(r3, r4),

            ResidualBlock(r4, output_size, True)
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        
        return x

