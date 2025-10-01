import torch
from torch import nn
from torchvision import models as models
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.transforms import *
import torch.onnx


class WeightedSumLoss(nn.Module):
    """Aggregate multiple loss functions in one weighted sum."""

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.values = {}

    def forward(self, outputs, **kwargs):
        total_loss = outputs.new(1).zero_()
        for loss in self.losses:
            loss_val = self.losses[loss](outputs=outputs, **kwargs)
            total_loss += self.weights[loss] * loss_val
            self.values[loss] = loss_val
        if self.normalize:
            total_loss /= sum(self.weights.values())
        return total_loss

    def add_loss(self, name, loss, weight=1.0):
        self.weights[name] = weight
        self.losses.add_module(name, loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
