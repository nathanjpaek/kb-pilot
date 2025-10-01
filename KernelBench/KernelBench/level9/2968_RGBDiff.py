import torch
from torch import nn
from torchvision import models as models
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx


class RGBDiff(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, image):
        """
        Args:
            image (torch.Tensor):  (N x T x C x H x W)

        """
        diffs = []
        for i in range(1, image.size(self.dim)):
            prev = image.index_select(self.dim, image.new_tensor(i - 1,
                dtype=torch.long))
            current = image.index_select(self.dim, image.new_tensor(i,
                dtype=torch.long))
            diffs.append(current - prev)
        return torch.cat(diffs, dim=self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
