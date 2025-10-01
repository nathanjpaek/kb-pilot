import torch
import torch.nn as nn
import torch.utils.data


class NormedMSE(nn.MSELoss):

    def forward(self, inp, tgt, *args, **kwargs):
        """
        Args:
            inp: (*, C)
            tgt: (*, C)
            Will normalize the input before the loss
        """
        inp = nn.functional.normalize(inp, dim=-1, p=2)
        tgt = nn.functional.normalize(tgt, dim=-1, p=2)
        return super().forward(inp, tgt, *args, **kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
