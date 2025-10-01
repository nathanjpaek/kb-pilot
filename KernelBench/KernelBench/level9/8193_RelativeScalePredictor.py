import torch
from torch import nn


class RelativeScalePredictor(nn.Module):

    def __init__(self, in_size, out_size):
        super(RelativeScalePredictor, self).__init__()
        self.predictor = nn.Linear(in_size, out_size)

    def forward(self, feat):
        predictions = self.predictor.forward(feat) + 1
        predictions = torch.nn.functional.relu(predictions) + 1e-12
        return predictions.log()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
