import math
import torch
import torchvision.transforms.functional as F
from torch import nn
from torch.nn import functional as F


class ArcMarginProduct(nn.Module):
    """ Process the latent vectors to output the cosine vector 
    for the follow-up ArcFaceLoss computation.

    Args:
        in_features: the column dimension of the weights,
            which is identical to the dim of latent vectors.
        out_features: the row dimension of the weights,
            which is identical to the number of classes.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)
            )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
