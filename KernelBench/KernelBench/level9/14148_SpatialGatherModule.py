import torch
import torch.nn.functional as F
import torch.nn as nn


class SpatialGatherModule(nn.Module):

    def __init__(self, scale=1, **kwargs):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale
    """forward"""

    def forward(self, features, probs):
        batch_size, num_classes, _h, _w = probs.size()
        probs = probs.view(batch_size, num_classes, -1)
        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, features)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
