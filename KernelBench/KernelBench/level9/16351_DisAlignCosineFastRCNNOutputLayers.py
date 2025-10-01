import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from itertools import product as product
from math import sqrt as sqrt
import torch.nn


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class NormalizedLinear(torch.nn.Module):
    """
    A advanced Linear layer which supports weight normalization or cosine normalization.

    """

    def __init__(self, in_features, out_features, bias=False, feat_norm=
        True, scale_mode='learn', scale_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.scale_mode = scale_mode
        self.scale_init = scale_init
        self.weight = torch.nn.Parameter(torch.Tensor(out_features,
            in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if self.scale_mode == 'constant':
            self.scale = scale_init
        elif self.scale_mode == 'learn':
            self.scale = torch.nn.Parameter(torch.ones(1) * scale_init)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight
                )
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): (N, C)
        Return:
            output (torch.Tensor): (N, D)
        """
        if self.feat_norm:
            inputs = torch.nn.functional.normalize(inputs, dim=1)
        output = inputs.mm(torch.nn.functional.normalize(self.weight, dim=1
            ).t())
        output = self.scale * output
        return output

    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}'
        if self.bias is None:
            s += ', bias=False'
        s += ', feat_norm={feat_norm}'
        s += ', scale_mode={scale_mode}'
        s += ', scale_init={scale_init}'
        return s.format(**self.__dict__)


class DisAlignCosineFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg,
        box_dim=4, scale_mode='learn', scale_init=20.0):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(DisAlignCosineFastRCNNOutputLayers, self).__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.cls_score = NormalizedLinear(input_size, num_classes + 1,
            scale_mode=scale_mode, scale_init=scale_init)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        self.logit_scale = nn.Parameter(torch.ones(num_classes))
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))
        self.confidence_layer = nn.Linear(input_size, 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.confidence_layer.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.confidence_layer, self.bbox_pred]:
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        confidence = self.confidence_layer(x).sigmoid()
        scores_tmp = confidence * (scores[:, :-1] * self.logit_scale + self
            .logit_bias)
        scores_tmp = scores_tmp + (1 - confidence) * scores[:, :-1]
        aligned_scores = cat([scores_tmp, scores[:, -1].view(-1, 1)], dim=1)
        proposal_deltas = self.bbox_pred(x)
        return aligned_scores, proposal_deltas


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_classes': 4, 'cls_agnostic_bbox_reg': 4}
        ]
