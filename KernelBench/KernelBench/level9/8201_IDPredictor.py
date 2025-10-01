import torch
import torch.nn.functional as F
from torch import nn


class IDPredictor(nn.Module):

    def __init__(self, nz_feat, n_dim=5):
        super(IDPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 256)
        self.sc_layer = nn.Linear(256, 128)
        self.sc_layer2 = nn.Linear(128, 64)

    def forward(self, feat):
        pred = self.pred_layer.forward(feat)
        pred = F.relu(pred)
        pred = self.sc_layer.forward(pred)
        pred = F.relu(pred)
        pred = self.sc_layer2.forward(pred)
        return pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nz_feat': 4}]
