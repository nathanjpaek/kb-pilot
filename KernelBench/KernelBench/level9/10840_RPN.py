import torch
from torch.nn import functional as F
from torch import nn
from torchvision import models as models
import torch.onnx


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True,
    dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class RPN(nn.Module):

    def __init__(self, dim_in, dim_internal, priors_num,
        class_activation_mode='softmax'):
        super().__init__()
        assert class_activation_mode in ('softmax', 'sigmoid')
        self.dim_in = dim_in
        self.dim_internal = dim_internal
        self.priors_num = priors_num
        self.dim_score = (priors_num * 2 if class_activation_mode ==
            'softmax' else priors_num)
        self.class_activation_mode = class_activation_mode
        self.conv = nn.Conv2d(dim_in, dim_internal, 3, 1, 1)
        self.cls_score = nn.Conv2d(dim_internal, self.dim_score, 1, 1, 0)
        self.bbox_deltas = nn.Conv2d(dim_internal, 4 * priors_num, 1, 1, 0)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_deltas.weight, std=0.01)
        nn.init.constant_(self.bbox_deltas.bias, 0)

    def forward(self, x):
        conv = F.relu(self.conv(x), inplace=True)
        cls_scores = self.cls_score(conv)
        bbox_deltas = self.bbox_deltas(conv)
        if self.class_activation_mode == 'softmax':
            b, _c, h, w = cls_scores.shape
            cls_scores = cls_scores.view(b, 2, -1, h, w)
            cls_probs = F.softmax(cls_scores, dim=1)[:, 1].squeeze(dim=1)
        else:
            cls_probs = torch.sigmoid(cls_scores)
        return bbox_deltas, cls_scores, cls_probs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_internal': 4, 'priors_num': 4}]
