import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class DQNLoss(_Loss):

    def __init__(self, mode='huber', size_average=None, reduce=None,
        reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.mode = mode
        self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[mode]

    def forward(self, nn_outputs, actions, target_outputs):
        target = nn_outputs.clone().detach()
        target[np.arange(target.shape[0]), actions] = target_outputs
        return self.loss(nn_outputs, target, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
