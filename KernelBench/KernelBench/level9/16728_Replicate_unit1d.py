import torch


class Replicate_unit1d(torch.nn.Module):

    def __init__(self, width, height):
        super(Replicate_unit1d, self).__init__()
        self.width = width
        self.height = height

    def forward(self, x):
        assert len(x.size()) == 2
        batch_num = x.size()[0]
        tmp = torch.cat([x.view((batch_num, -1, 1)) for _ in range(self.
            width)], dim=2)
        ret = torch.cat([tmp.view((batch_num, tmp.size()[1], tmp.size()[2],
            1)) for _ in range(self.height)], dim=3)
        return ret


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'width': 4, 'height': 4}]
