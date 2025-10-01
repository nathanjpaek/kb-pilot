import torch
import torch.nn.parallel
import torch.utils.data
import torch.distributions


class CompositionFunction(torch.nn.Module):

    def __init__(self, representation_size: 'int'):
        super().__init__()

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError


class MLPComposition(CompositionFunction):

    def __init__(self, representation_size: 'int'):
        super().__init__(representation_size)
        self.linear_1 = torch.nn.Linear(representation_size * 2, 50)
        self.linear_2 = torch.nn.Linear(50, representation_size)

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        return self.linear_2(torch.tanh(self.linear_1(torch.cat((x, y), dim
            =1))))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'representation_size': 4}]
