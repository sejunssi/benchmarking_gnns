import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    def __index__(self):
        super(LabelSmoothingLoss, self).__init__()

    def forward(self, pred, label):
        loss = (- label * nn.LogSoftmax(dim=1)(pred))
        loss = loss.sum(dim=0)
        return torch.mean(loss)


