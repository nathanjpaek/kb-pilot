import torch
import torch.nn as nn


class KMaxPool1d(nn.Module):

    def __init__(self, top_k: 'int'):
        super(KMaxPool1d, self).__init__()
        self.top_k = top_k

    def forward(self, inputs):
        assert inputs.dim() == 3
        top_idxs = torch.topk(inputs, k=self.top_k, dim=2)[1]
        sorted_top_idxs = top_idxs.sort(dim=2)[0]
        return inputs.gather(dim=2, index=sorted_top_idxs)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'top_k': 4}]
