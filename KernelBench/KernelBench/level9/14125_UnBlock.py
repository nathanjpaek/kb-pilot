import torch
import torch.nn as nn
import torch.utils.cpp_extension


def unblock(tensor):
    """blocked tensor back to normal"""
    B, M, N, C = tensor.size()
    H = W = int(M ** 0.5)
    patch_size = int(N ** 0.5)
    tensor = tensor.reshape(B, H, W, patch_size, patch_size, C)
    tensor = tensor.permute(0, 5, 3, 1, 4, 2).reshape(B, C, H * patch_size,
        W * patch_size)
    return tensor


class UnBlock(nn.Module):

    def forward(self, x):
        return unblock(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
