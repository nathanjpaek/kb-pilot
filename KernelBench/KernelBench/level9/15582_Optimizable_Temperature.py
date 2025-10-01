import torch
import torch.utils.data


class Optimizable_Temperature(torch.nn.Module):

    def __init__(self, initial_temperature=None):
        super(Optimizable_Temperature, self).__init__()
        self.log_temperature = torch.nn.Parameter(data=torch.zeros([1]).
            type(torch.DoubleTensor))
        if initial_temperature is not None:
            self.log_temperature.data = torch.log(torch.tensor(
                initial_temperature).type(torch.DoubleTensor))

    def forward(self):
        return torch.exp(self.log_temperature)


def get_inputs():
    return []


def get_init_inputs():
    return [[], {}]
