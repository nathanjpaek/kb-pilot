import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn


def mask_logits(inputs, mask, mask_value=-1e+30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class WeightedPool(nn.Module):

    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
