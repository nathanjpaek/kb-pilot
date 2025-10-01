import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLossBlock(nn.Module):

    def __init__(self, target: 'torch.Tensor'):
        super().__init__()
        self.stored_value = None
        self._loss = F.mse_loss
        self.shape = target.shape
        self._target_gram_matrix = nn.Parameter(self.gram_matrix(target).data)

    @staticmethod
    def gram_matrix(x: 'torch.Tensor') ->torch.Tensor:
        bs, ch, h, w = x.size()
        f = x.view(bs, ch, w * h)
        f_t = f.transpose(1, 2)
        g = f.bmm(f_t) / (ch * h * w)
        return g

    def forward(self, input_tensor: 'torch.Tensor') ->torch.Tensor:
        input_gram_matrix = self.gram_matrix(input_tensor)
        result = self._loss(input_gram_matrix, self._target_gram_matrix)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'target': torch.rand([4, 4, 4, 4])}]
