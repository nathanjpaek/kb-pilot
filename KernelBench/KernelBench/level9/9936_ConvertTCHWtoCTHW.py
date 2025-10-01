import torch
import torch.utils.data


class ConvertTCHWtoCTHW(torch.nn.Module):
    """
    Convert a torch.FloatTensor of shape (TIME x CHANNELS x HEIGHT x WIDTH) to
    a torch.FloatTensor of shape (CHANNELS x TIME x HEIGHT x WIDTH).
    """

    def forward(self, tensor):
        return tensor.permute(1, 0, 2, 3).contiguous()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
