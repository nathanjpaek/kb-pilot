import torch
from torch import nn


class DiceCoefMultilabelLoss(nn.Module):

    def __init__(self, cuda=True):
        super().__init__()
        self.one = torch.tensor(1.0, dtype=torch.float32)
        self.activation = torch.nn.Softmax2d()

    def dice_loss(self, predict, target):
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = predict * target.float()
        score = (intersection.sum() * 2.0 + 1.0) / (predict.sum() + target.
            sum() + 1.0)
        return 1.0 - score

    def forward(self, predict, target, numLabels=3, channel='channel_first'):
        assert channel == 'channel_first' or channel == 'channel_last', "channel has to be either 'channel_first' or 'channel_last'"
        dice = 0
        predict = self.activation(predict)
        if channel == 'channel_first':
            for index in range(numLabels):
                temp = self.dice_loss(predict[:, index, :, :], target[:,
                    index, :, :])
                dice += temp
        else:
            for index in range(numLabels):
                temp = self.dice_loss(predict[:, :, :, index], target[:, :,
                    :, index])
                dice += temp
        dice = dice / numLabels
        return dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
