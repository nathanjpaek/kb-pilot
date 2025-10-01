import torch
from torch import nn


class GDL(nn.Module):

    def __init__(self, drop_rate=0.8, drop_th=0.7):
        super(GDL, self).__init__()
        if not 0 <= drop_rate <= 1:
            raise ValueError('drop-rate must be in range [0, 1].')
        if not 0 <= drop_th <= 1:
            raise ValueError('drop-th must be in range [0, 1].')
        self.drop_rate = drop_rate
        self.drop_th = drop_th
        self.attention = None
        self.drop_mask = None

    def forward(self, input_):
        attention = torch.mean(input_, dim=1, keepdim=True)
        importance_map = torch.sigmoid(attention)
        drop_mask = self._drop_mask(attention)
        selected_map = self._select_map(importance_map, drop_mask)
        return (input_.mul(selected_map) + input_) / 2

    def _select_map(self, importance_map, drop_mask):
        random_tensor = torch.rand([], dtype=torch.float32) + self.drop_rate
        binary_tensor = random_tensor.floor()
        return (1.0 - binary_tensor
            ) * importance_map + binary_tensor * drop_mask

    def _drop_mask(self, attention):
        b_size = attention.size(0)
        max_val, _ = torch.max(attention.view(b_size, -1), dim=1, keepdim=True)
        thr_val = max_val * self.drop_th
        thr_val = thr_val.view(b_size, 1, 1, 1)
        return (attention < thr_val).float()

    def extra_repr(self):
        return 'drop_rate={}, drop_th={}'.format(self.drop_rate, self.drop_th)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
