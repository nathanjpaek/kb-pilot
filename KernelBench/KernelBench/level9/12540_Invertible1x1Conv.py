import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.nn


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=
            0, bias=False)
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        W = W.contiguous()
        self.conv.weight.data = W

    def forward(self, z):
        batch_size, _group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        log_det_W = batch_size * n_of_groups * torch.logdet(W.unsqueeze(0).
            float()).squeeze()
        z = self.conv(z)
        return z, log_det_W

    def infer(self, z):
        _batch_size, _group_size, _n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        if not hasattr(self, 'W_inverse'):
            W_inverse = W.float().inverse()
            W_inverse = Variable(W_inverse[..., None])
            if z.type() == 'torch.cuda.HalfTensor' or z.type(
                ) == 'torch.HalfTensor':
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse
        z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'c': 4}]
