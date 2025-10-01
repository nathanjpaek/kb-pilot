import torch
import torch.nn as nn


class MultiscalePixelLoss(nn.Module):

    def __init__(self, loss_f=nn.L1Loss(), scale=5):
        super(MultiscalePixelLoss, self).__init__()
        self.criterion = loss_f
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor', mask=None
        ) ->torch.Tensor:
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, x.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:
                loss += self.weights[i] * self.criterion(x * mask, y * mask)
            else:
                loss += self.weights[i] * self.criterion(x, y)
            if i != len(self.weights) - 1:
                x = self.downsample(x)
                y = self.downsample(y)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 64, 64]), torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]
