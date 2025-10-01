import torch
import torch.utils.data


class ResNNFlow(torch.nn.Sequential):

    def __init__(self, *args, **kwargs):
        super(ResNNFlow, self).__init__(*args, **kwargs)
        self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))

    def forward(self, inputs):
        or_inputs = inputs
        for module in self._modules.values():
            inputs = module(inputs)
        return self.gate.sigmoid() * inputs + (1 - self.gate.sigmoid()
            ) * or_inputs

    def logdetj(self, inputs=None):
        for module in self._modules.values():
            inputs = module.log_diag_jacobian(inputs)
            inputs = inputs if len(inputs.shape) == 4 else inputs.view(
                inputs.shape + [1, 1])
        return (torch.nn.functional.softplus(grad.squeeze() + self.gate) -
            torch.nn.functional.softplus(self.gate)).sum(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
