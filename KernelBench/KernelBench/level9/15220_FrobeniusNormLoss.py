import torch
import torch.nn as nn


def get_outnorm(x: 'torch.Tensor', out_norm: 'str'='') ->torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    img_shape = x.shape
    if not out_norm:
        return 1
    norm = 1
    if 'b' in out_norm:
        norm /= img_shape[0]
    if 'c' in out_norm:
        norm /= img_shape[-3]
    if 'i' in out_norm:
        norm /= img_shape[-1] * img_shape[-2]
    return norm


class FrobeniusNormLoss(nn.Module):

    def __init__(self, order='fro', out_norm: 'str'='c', kind: 'str'='vec'):
        super().__init__()
        self.order = order
        self.out_norm = out_norm
        self.kind = kind

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        norm = get_outnorm(x, self.out_norm)
        if self.kind == 'mat':
            loss = torch.linalg.matrix_norm(x - y, ord=self.order).mean()
        else:
            loss = torch.linalg.norm(x.view(-1, 1) - y.view(-1, 1), ord=
                self.order)
        return loss * norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
