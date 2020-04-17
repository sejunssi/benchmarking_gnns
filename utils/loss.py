import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    def __index__(self):
        super(LabelSmoothingLoss, self).__init__()

    def forward(self, pred, label):
        return torch.mean((-label * nn.LogSoftmax(dim=1)(pred)).mean(dim=0))


