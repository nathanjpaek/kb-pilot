import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.init


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt().view(X.size(0), -1)
    X = torch.div(X, norm.expand_as(X))
    return X


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in list(state_dict.items()):
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecomp, self).load_state_dict(new_state)

    def __call__(self, images):
        return self.forward(images)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'img_dim': 4, 'embed_size': 4}]
