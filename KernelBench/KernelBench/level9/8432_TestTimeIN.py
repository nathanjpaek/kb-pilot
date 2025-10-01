import torch
import torch.nn as nn
import torch.optim
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed


class TestTimeIN(nn.BatchNorm2d):

    def __init__(self, num_features: 'int', eps: 'float'=1e-05, momentum:
        'float'=1, affine: 'bool'=True, track_running_stats: 'bool'=True):
        super().__init__(num_features, eps, momentum, affine,
            track_running_stats)

    def forward(self, target_input):
        target_input.numel() / target_input.size(1)
        with torch.no_grad():
            target_instance_var = target_input.var([2, 3], unbiased=False)[
                :, :, None, None]
            target_instance_mean = target_input.mean([2, 3])[:, :, None, None]
            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]
            target_input = weight * (target_input - target_instance_mean
                ) / torch.sqrt(target_instance_var + self.eps) + bias
            target_input = torch.clamp(target_input, max=1)
            return target_input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
