from torch.nn import Module
import math
import torch
import torch.distributed
import torch.nn.functional as F


class ArcFaceLinear(Module):

    def __init__(self, embedding_size, num_classes):
        super(ArcFaceLinear, self).__init__()
        self.weight = torch.nn.Parameter(data=torch.FloatTensor(num_classes,
            embedding_size), requires_grad=True)
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(features, F.normalize(self.weight))
        return cosine


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4, 'num_classes': 4}]
