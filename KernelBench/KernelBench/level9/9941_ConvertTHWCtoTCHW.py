import torch
import torch.utils.data


class ConvertTHWCtoTCHW(torch.nn.Module):
    """
    Convert a torch.FloatTensor of shape (TIME x HEIGHT x WIDTH x CHANNEL) to
    a torch.FloatTensor of shape (TIME x CHANNELS x HEIGHT x WIDTH).
    """

    def forward(self, tensor):
        return tensor.permute(0, 3, 1, 2).contiguous()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
