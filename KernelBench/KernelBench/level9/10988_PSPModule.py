import torch
import torch.nn as nn
import torch._C
import torch.serialization


class PSPModule(nn.Module):
    """Reference: https://github.com/MendelXu/ANN
    """
    methods = {'max': nn.AdaptiveMaxPool2d, 'avg': nn.AdaptiveAvgPool2d}

    def __init__(self, sizes=(1, 3, 6, 8), method='max'):
        super().__init__()
        assert method in self.methods
        pool_block = self.methods[method]
        self.stages = nn.ModuleList([pool_block(output_size=(size, size)) for
            size in sizes])

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        out = torch.cat(priors, -1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
