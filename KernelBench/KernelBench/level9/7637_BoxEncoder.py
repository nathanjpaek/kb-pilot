import torch
from torch import nn
import torch.utils.data


class BoxEncoder(nn.Module):

    def __init__(self, boxSize, featureSize, hiddenSize):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(boxSize, featureSize)
        self.middlein = nn.Linear(featureSize, hiddenSize)
        self.middleout = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, boxes_in):
        boxes = self.encoder(boxes_in)
        boxes = self.tanh(boxes)
        boxes = self.middlein(boxes)
        boxes = self.tanh(boxes)
        boxes = self.middleout(boxes)
        boxes = self.tanh(boxes)
        return boxes


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'boxSize': 4, 'featureSize': 4, 'hiddenSize': 4}]
