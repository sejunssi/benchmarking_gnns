import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class ResnetMLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        list_conv_layers = [self.conv1x1(input_dim, input_dim // 2 ** (l + 1)) for l in range(L)]
        list_conv_layers.append(self.conv1x1(input_dim, output_dim))
        self.downsample_conv = nn.ModuleList(list_conv_layers)
        list_bn_layers = [nn.BatchNorm2d(input_dim // 2 ** (l + 1)) for l in range(L)]
        list_bn_layers.append(nn.BatchNorm2d(output_dim))
        self.downsample_bn = nn.ModuleList(list_bn_layers)
        self.L = L

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            if l is not 0:
                if self.downsample_conv is not None:
                    identity = self.downsample_conv[l](x.reshape((x.shape[0], x.shape[1], 1, 1)))
                    identity = self.downsample_bn[l](identity).squeeze()
                y += identity
            y = F.relu(y)

        y = self.FC_layers[self.L](y)
        identity = self.downsample_conv[self.L](x.reshape((x.shape[0], x.shape[1], 1, 1)))
        identity = self.downsample_bn[self.L](identity).squeeze(3).squeeze(2)
        y = y+identity
        return y