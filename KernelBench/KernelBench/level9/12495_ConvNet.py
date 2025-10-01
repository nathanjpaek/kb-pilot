import torch
import torch.nn
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.cuda
import torch.cuda.nccl
import torch.backends.cudnn
import torch.backends.mkl


class ConvNet(nn.Module):

    def __init__(self, gpus, layouts, dtypes):
        super(ConvNet, self).__init__()
        self.dtypes = dtypes
        if isinstance(gpus, list):
            self.layer_gpus = gpus
        else:
            gpus = [gpus] * 4
        self.conv0 = torch.nn.Conv2d(8, 16, (2, 2))
        self.conv1 = torch.nn.Conv2d(16, 32, (2, 2))
        self.conv2 = torch.nn.Conv2d(32, 16, (2, 2))
        self.conv3 = torch.nn.Conv2d(16, 8, (2, 2))

    def forward(self, x):
        x = x
        self.layer_gpus if hasattr(self, 'layer_gpus') else [x.device] * 4
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


def get_inputs():
    return [torch.rand([4, 8, 64, 64])]


def get_init_inputs():
    return [[], {'gpus': False, 'layouts': 4, 'dtypes': torch.float32}]
