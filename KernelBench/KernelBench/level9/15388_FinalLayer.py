import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class normrelu(nn.Module):

    def __init__(self):
        super(normrelu, self).__init__()

    def forward(self, x):
        dim = 1
        x = F.relu(x) / torch.max(x, dim, keepdim=True)[0]
        return x


class LinearBlock(nn.Module):

    def __init__(self, pre_dim, dim, activation='none', dropout_rate=0,
        use_batch_norm=False, use_layer_norm=False):
        self.linear = None
        self.bn = None
        self.ln = None
        self.act = None
        self.dropout_layer = None
        super(LinearBlock, self).__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'normrelu':
            self.act = normrelu()
        elif activation == 'none':
            self.act = None
        else:
            None
        if use_batch_norm:
            self.linear = nn.Linear(pre_dim, dim, bias=False)
            self.bn = nn.BatchNorm1d(dim, momentum=0.05)
        else:
            self.linear = nn.Linear(pre_dim, dim)
        if use_layer_norm:
            self.ln = LayerNorm(dim)
        if dropout_rate > 0.0001:
            self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        if self.linear is not None:
            x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.ln is not None:
            x = self.ln(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class FinalLayer(nn.Module):
    """
    final classification and bounding box regression layer for RPN KWS
    """

    def __init__(self, input_dim, num_class):
        super(FinalLayer, self).__init__()
        self.linear = LinearBlock(input_dim, input_dim, activation='relu')
        self.cls_score_KWS = nn.Linear(input_dim, num_class, bias=True)
        self.bbox_score_KWS = nn.Linear(input_dim, 2, bias=True)

    def forward(self, x):
        x = self.linear(x)
        kws_cls_score = self.cls_score_KWS(x)
        kws_bbox_pred = self.bbox_score_KWS(x)
        return kws_cls_score, kws_bbox_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'num_class': 4}]
