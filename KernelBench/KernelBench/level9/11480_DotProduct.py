import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn


class DotProduct(nn.Module):

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        """
        Inputs:
            x - (N, F)
            y - (N, F)
        Output:
            output - (N, 1) dot-product output
        """
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        assert x.shape == y.shape
        x = nn.functional.normalize(x, dim=1)
        y = nn.functional.normalize(y, dim=1)
        output = torch.matmul(x.unsqueeze(1), y.unsqueeze(2)).squeeze(2)
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
