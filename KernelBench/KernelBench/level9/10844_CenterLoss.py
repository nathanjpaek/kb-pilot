import torch
from torch.nn import functional as F
from torch import nn
from torchvision import models as models
import torch.onnx


class CenterLoss(nn.Module):
    """Implements the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""

    def __init__(self, num_classes, embed_size, cos_dist=True):
        super().__init__()
        self.cos_dist = cos_dist
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, embed_size))
        self.embed_size = embed_size
        self.mse = nn.MSELoss(reduction='elementwise_mean')

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def forward(self, features, labels):
        features = F.normalize(features)
        batch_size = labels.size(0)
        features_dim = features.size(1)
        assert features_dim == self.embed_size
        if self.cos_dist:
            self.centers.data = F.normalize(self.centers.data, p=2, dim=1)
        centers_batch = self.centers[labels, :]
        if self.cos_dist:
            cos_sim = nn.CosineSimilarity()
            cos_diff = 1.0 - cos_sim(features, centers_batch)
            center_loss = torch.sum(cos_diff) / batch_size
        else:
            center_loss = self.mse(centers_batch, features)
        return center_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'num_classes': 4, 'embed_size': 4}]
