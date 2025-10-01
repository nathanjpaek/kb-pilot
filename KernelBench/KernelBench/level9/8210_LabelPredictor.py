import torch
from torch import nn


class LabelPredictor(nn.Module):

    def __init__(self, nz_feat, classify_rot=True):
        super(LabelPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 1)

    def forward(self, feat):
        pred = self.pred_layer.forward(feat)
        pred = torch.sigmoid(pred)
        return pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nz_feat': 4}]
