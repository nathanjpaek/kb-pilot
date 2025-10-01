import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn


class JointsDistLoss(nn.Module):

    def __init__(self):
        super(JointsDistLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        batch = output.size(0)
        output.size(1)
        output = output.reshape(batch, 32, 2)
        target = target.reshape(batch, 32, 2)
        loss = self.criterion(output[:, :, 0], target[:, :, 0]
            ) + 0.3 * self.criterion(output[:, :, 1], target[:, :, 1])
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
