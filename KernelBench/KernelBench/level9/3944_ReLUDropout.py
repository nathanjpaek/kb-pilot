import torch
import torch.utils.data
import torch.cuda
import torch.utils.checkpoint


def relu_dropout(x, p=0, training=False, variational=False, batch_first=False):
    if not training or p == 0:
        return x.clamp_(min=0)
    p1m = 1 - p
    if variational:
        if batch_first:
            mask = torch.rand_like(x[:, 0, :]) > p1m
            mask = mask.unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            mask = torch.rand_like(x[0]) > p1m
            mask = mask.unsqueeze(0).repeat(x.size(0), 1, 1)
    else:
        mask = torch.rand_like(x) > p1m
    mask |= x < 0
    return x.masked_fill_(mask, 0).div_(p1m)


class ReLUDropout(torch.nn.Dropout):

    def __init__(self, p=0.5, variational=False, batch_first=False, inplace
        =False):
        super().__init__(p, inplace=True)
        self.variational = variational
        self.batch_first = batch_first

    def forward(self, input):
        return relu_dropout(input, p=self.p, training=self.training,
            variational=self.variational, batch_first=self.batch_first)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
