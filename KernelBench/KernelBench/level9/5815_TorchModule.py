import torch
import torch.nn


class TorchLinearModule(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super(TorchLinearModule, self).__init__()
        self._linear = torch.nn.Linear(in_size, out_size)

    def forward(self, x):
        return self._linear(x)


class TorchModule(torch.nn.Module):

    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        super(TorchModule, self).__init__()
        self._linear0 = TorchLinearModule(in_size, hidden_size)
        self._linear1 = TorchLinearModule(hidden_size, hidden_size)
        self._linear2 = TorchLinearModule(hidden_size, out_size)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = torch.tanh(self._linear0(x))
        x = torch.tanh(self._linear1(x))
        return torch.tanh(self._linear2(x))[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
