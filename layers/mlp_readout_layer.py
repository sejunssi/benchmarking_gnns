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
        list_FC_layers = [nn.Linear(input_dim , input_dim, bias=True) for _ in range(L)]
        list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        identity = y
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y + identity)
            identity = y
        y = self.FC_layers[self.L](y)
        return y

class RK2netMLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim , input_dim, bias=True) for _ in range(L)]
        list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)

        list_FC_layers2 = [nn.Linear(input_dim, input_dim, bias=True) for _ in range(L)]
        self.FC_layers2 = nn.ModuleList(list_FC_layers2)

        self.L = L

    def forward(self, x):
        y = x
        identity = y
        for l in range(self.L):
            y1 = self.FC_layers[l](y)
            y1 = y1 + identity
            y2 = self.FC_layers2[l](y1)
            y = F.relu(y1 + y2)
            identity = y
        y = self.FC_layers[self.L](y)
        return y


class BaseLineMLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2, i=0):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim, input_dim, bias=True) for _ in range(L)]
        list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class RKinetMLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2, i=0):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim , input_dim, bias=True) for _ in range(L)]
        list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        if i == 2:
            list_FC_layers2 = [RK2netMLPReadout(input_dim, input_dim, L) for _ in range(L)]
            list_FC_layers2.append(ResnetMLPReadout(input_dim, output_dim, L))
        elif i == 3:
            list_FC_layers2 = [RK3netMLPReadout(input_dim, input_dim) for _ in range(L)]
            list_FC_layers2.append(ResnetMLPReadout(input_dim, output_dim, L))
        else:
            list_FC_layers2 = [ResnetMLPReadout(input_dim, input_dim, L) for _ in range(L)]
            list_FC_layers2.append(ResnetMLPReadout(input_dim, output_dim, L))
        self.FC_layers2 = nn.ModuleList(list_FC_layers2)
        self.L = L

    def forward(self, x):
        y1 = x
        y2 = x
        identity = y1
        for l in range(self.L):
            y1 = self.FC_layers[l](y1)
            F.relu(y1)
        y1 = self.FC_layers[self.L](y1)
        for l in range(self.L):
            y2 = self.FC_layers2[l](y2)
            y2 = F.relu(identity + y2)
            identity = y2
        y2 = self.FC_layers[self.L](y2)
        return y1+y2

class RK2M1netMLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim , input_dim, bias=True) for _ in range(L)]
        list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)

        list_FC_layers2 = [nn.Linear(input_dim, input_dim, bias=True) for _ in range(L)]
        self.FC_layers2 = nn.ModuleList(list_FC_layers2)

        self.L = L

    def forward(self, x):
        y = x
        identity = y
        for l in range(self.L-1):
            y1 = self.FC_layers[l](y)
            y1 = y1 + identity
            y2 = self.FC_layers2[l](y1)
            y = F.relu(y1 + y2)
            identity = y
        y = self.FC_layers[self.L](y)
        return y

class RK3netMLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim , input_dim, bias=True) for _ in range(L)]
        list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)

        list_FC_layers2 = [nn.Linear(input_dim, input_dim, bias=True) for _ in range(L)]
        self.FC_layers2 = nn.ModuleList(list_FC_layers2)

        list_FC_layers3 = [nn.Linear(input_dim, input_dim, bias=True) for _ in range(L)]
        self.FC_layers3 = nn.ModuleList(list_FC_layers3)

        self.L = L

    def forward(self, x):
        y = x
        identity = y
        for l in range(self.L):
            y1 = self.FC_layers[l](y)
            y1 = y1 + identity
            y2 = self.FC_layers2[l](y1)
            y2 = y1 + y2
            y3 = self.FC_layers2[l](y2)
            y = F.relu(y2 + y3)
            identity = y
        y = self.FC_layers[self.L](y)
        return y
#
#
# class ResnetMLPReadout(nn.Module):
#
#     def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
#         super().__init__()
#         list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
#         list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
#         self.FC_layers = nn.ModuleList(list_FC_layers)
#         list_conv_layers = [self.conv1x1(input_dim, input_dim // 2 ** (l + 1)) for l in range(L)]
#         list_conv_layers.append(self.conv1x1(input_dim, output_dim))
#         self.downsample_conv = nn.ModuleList(list_conv_layers)
#         list_bn_layers = [nn.BatchNorm2d(input_dim // 2 ** (l + 1)) for l in range(L)]
#         list_bn_layers.append(nn.BatchNorm2d(output_dim))
#         self.downsample_bn = nn.ModuleList(list_bn_layers)
#         self.L = L
#
#     def conv1x1(self, in_planes, out_planes, stride=1):
#         """1x1 convolution"""
#         return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#     def forward(self, x):
#         y = x
#         for l in range(self.L):
#             y = self.FC_layers[l](y)
#             if l is not 0:
#                 if self.downsample_conv is not None:
#                     identity = self.downsample_conv[l](x.reshape((x.shape[0], x.shape[1], 1, 1)))
#                     identity = self.downsample_bn[l](identity).squeeze()
#                 y += identity
#             y = F.relu(y)
#
#         y = self.FC_layers[self.L](y)
#         identity = self.downsample_conv[self.L](x.reshape((x.shape[0], x.shape[1], 1, 1)))
#         identity = self.downsample_bn[self.L](identity).squeeze(3).squeeze(2)
#         y = y+identity
#         return y