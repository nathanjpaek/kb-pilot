from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class my_BinaryCross(nn.Module):

    def __init__(self, args):
        super(my_BinaryCross, self).__init__()
        self.args = args

    def forward(self, output, target, beat):
        modif_beat = 1.0 / torch.exp(beat) * 10
        modif_beat[modif_beat < 7] = 5 / 100
        modif_beat[modif_beat > 7] = 5 / 100
        batch_size = len(output)
        len_pred = len(output[0])
        loss = -torch.mean(modif_beat * torch.sum(target.view(batch_size,
            len_pred, -1) * torch.log(output.view(batch_size, len_pred, -1)
            ), dim=2))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config()}]
