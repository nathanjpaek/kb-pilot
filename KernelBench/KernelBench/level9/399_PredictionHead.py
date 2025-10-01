import torch
import torch.nn as nn


class PredictionHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors):
        super(PredictionHead, self).__init__()
        self.classification = nn.Conv2d(in_channels, num_classes *
            num_anchors, kernel_size=1)
        self.regression = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1
            )
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        bs = x.shape[0]
        class_logits = self.classification(x)
        box_regression = self.regression(x)
        class_logits = class_logits.permute(0, 2, 3, 1).reshape(bs, -1,
            self.num_classes)
        box_regression = box_regression.permute(0, 2, 3, 1).reshape(bs, -1, 4)
        return class_logits, box_regression


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_classes': 4, 'num_anchors': 4}]
