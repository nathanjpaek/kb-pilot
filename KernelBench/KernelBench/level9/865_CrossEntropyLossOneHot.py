import torch
from torch import Tensor
from torch.nn.modules.loss import CrossEntropyLoss


class CrossEntropyLossOneHot(CrossEntropyLoss):
    EPS: 'int' = 1e-07

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        input = torch.clip(input, self.EPS, 1 - self.EPS)
        crossentropy = target * torch.log(input)
        crossentropy = -torch.sum(crossentropy, -1)
        return torch.sum(crossentropy)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
