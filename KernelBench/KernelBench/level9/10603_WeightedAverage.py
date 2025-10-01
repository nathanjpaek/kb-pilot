import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F


def find_local_patch(x, patch_size):
    N, _C, H, W = x.shape
    x_unfold = F.unfold(x, kernel_size=(patch_size, patch_size), padding=(
        patch_size // 2, patch_size // 2), stride=(1, 1))
    return x_unfold.view(N, x_unfold.shape[1], H, W)


class WeightedAverage(nn.Module):

    def __init__(self):
        super(WeightedAverage, self).__init__()

    def forward(self, x_lab, patch_size=3, alpha=1, scale_factor=1):
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        l = x_lab[:, 0:1, :, :]
        a = x_lab[:, 1:2, :, :]
        b = x_lab[:, 2:3, :, :]
        local_l = find_local_patch(l, patch_size)
        local_a = find_local_patch(a, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_difference_l = (local_l - l) ** 2
        correlation = nn.functional.softmax(-1 * local_difference_l / alpha,
            dim=1)
        return torch.cat((torch.sum(correlation * local_a, dim=1, keepdim=
            True), torch.sum(correlation * local_b, dim=1, keepdim=True)), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
