import torch
import torch.multiprocessing
import torch.utils.data


class GetMask(torch.nn.Module):
    """
    inputs: x:          any size
    outputs:mask:       same size as input x
    """

    def __init__(self, pad_idx=0):
        super(GetMask, self).__init__()
        self.pad_idx = pad_idx

    def forward(self, x):
        mask = torch.ne(x, self.pad_idx).float()
        return mask


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
