import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F


def find_local_patch(x, patch_size):
    N, _C, H, W = x.shape
    x_unfold = F.unfold(x, kernel_size=(patch_size, patch_size), padding=(
        patch_size // 2, patch_size // 2), stride=(1, 1))
    return x_unfold.view(N, x_unfold.shape[1], H, W)


class NonlocalWeightedAverage(nn.Module):

    def __init__(self):
        super(NonlocalWeightedAverage, self).__init__()

    def forward(self, x_lab, feature, patch_size=3, alpha=0.1, scale_factor=1):
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        batch_size, _channel, height, width = x_lab.shape
        feature = F.interpolate(feature, size=(height, width))
        batch_size = x_lab.shape[0]
        x_ab = x_lab[:, 1:3, :, :].view(batch_size, 2, -1)
        x_ab = x_ab.permute(0, 2, 1)
        local_feature = find_local_patch(feature, patch_size)
        local_feature = local_feature.view(batch_size, local_feature.shape[
            1], -1)
        correlation_matrix = torch.matmul(local_feature.permute(0, 2, 1),
            local_feature)
        correlation_matrix = nn.functional.softmax(correlation_matrix /
            alpha, dim=-1)
        weighted_ab = torch.matmul(correlation_matrix, x_ab)
        weighted_ab = weighted_ab.permute(0, 2, 1).contiguous()
        weighted_ab = weighted_ab.view(batch_size, 2, height, width)
        return weighted_ab


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
