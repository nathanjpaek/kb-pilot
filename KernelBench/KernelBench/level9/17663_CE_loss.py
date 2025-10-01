import torch
import torch.nn as nn
import torch.utils.model_zoo


class CE_loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        n, _c, h, w = target.data.shape
        predict = predict.permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        target = target.permute(0, 2, 3, 1).contiguous().view(n * h * w, -1)
        return self.loss(predict, torch.max(target, 1)[1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
