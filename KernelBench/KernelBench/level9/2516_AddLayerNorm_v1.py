import torch
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class AddLayerNorm_v1(torch.nn.Module):

    def __init__(self, dim=32):
        super(AddLayerNorm_v1, self).__init__()
        self.layernorm = torch.nn.LayerNorm(dim)

    def forward(self, x, y, z):
        x = x + y + z
        return self.layernorm(x)


def get_inputs():
    return [torch.rand([4, 4, 32, 32]), torch.rand([4, 4, 32, 32]), torch.
        rand([4, 4, 32, 32])]


def get_init_inputs():
    return [[], {}]
