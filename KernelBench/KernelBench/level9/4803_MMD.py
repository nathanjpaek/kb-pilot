import torch
from torch import nn


class MMD(nn.Module):

    def __init__(self):
        super().__init__()

    def _guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5,
        fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.
            size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.
            size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 -
                n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [(bandwidth * kernel_mul ** i) for i in range(
            kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for
            bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target, kernel_mul=2.0, kernel_num=5,
        fix_sigma=None):
        source_num = int(source.size()[0])
        target_num = int(target.size()[0])
        kernels = self._guassian_kernel(source, target, kernel_mul=
            kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = torch.mean(kernels[:source_num, :source_num])
        YY = torch.mean(kernels[target_num:, target_num:])
        XY = torch.mean(kernels[:target_num, source_num:])
        YX = torch.mean(kernels[source_num:, :target_num])
        loss = XX + YY - XY - YX
        return loss

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult': 10}]


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
