import torch
import numpy as np
import torch.nn as nn
import torch.nn.init


def l2norm(matrix, dim, eps=1e-08):
    norm = torch.pow(matrix, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    matrix = matrix / norm
    return matrix


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_size, embed_size, use_abs=False, img_norm=True):
        super(EncoderImagePrecomp, self).__init__()
        self.use_abs = use_abs
        self.img_norm = img_norm
        self.fc = nn.Linear(img_size, embed_size)
        self.init_weights()

    def init_weights(self):
        """
        Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, img_features):
        """
        :param img_features: (batch_size, num_regions, row_img_features)
        :return: features: (batch_size, num_regions, img_features)
        """
        features = self.fc(img_features)
        if self.img_norm:
            features = l2norm(features, -1)
        if self.use_abs:
            features = torch.abs(features)
        return features


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'img_size': 4, 'embed_size': 4}]
