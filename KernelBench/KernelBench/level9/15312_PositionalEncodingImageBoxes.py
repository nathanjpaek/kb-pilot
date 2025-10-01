import torch
from torch import nn as nn
import torch.nn.init
from torchvision import models as models


class PositionalEncodingImageBoxes(nn.Module):

    def __init__(self, d_model, mode='project-and-sum'):
        super().__init__()
        self.mode = mode
        if mode == 'project-and-sum':
            self.map = nn.Linear(5, d_model)
        elif mode == 'concat-and-process':
            self.map = nn.Sequential(nn.Linear(d_model + 5, d_model), nn.
                ReLU(), nn.Linear(d_model, d_model))

    def forward(self, x, boxes):
        bs = x.shape[1]
        area = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[
            :, :, 1])
        area = area.unsqueeze(2)
        s_infos = torch.cat([boxes, area], dim=2)
        if self.mode == 'project-and-sum':
            ct = self.map(s_infos).permute(1, 0, 2)
            x = x + ct.expand(-1, bs, -1)
        elif self.mode == 'concat-and-process':
            x = torch.cat([x, s_infos.permute(1, 0, 2)], dim=2)
            x = self.map(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
