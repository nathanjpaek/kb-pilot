import torch


class BPR(torch.nn.Module):

    def __init__(self):
        super(BPR, self).__init__()
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, pos, neg):
        loss = torch.log(self._sigmoid(pos.double() - neg.double()))
        return -loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
