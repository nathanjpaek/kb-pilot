import torch
from torch import nn


class NN_2layer_regression(nn.Module):

    def __init__(self, input_dim, interm_dim1, interm_dim2):
        super().__init__()
        self.d = input_dim
        self.interm_dim1 = interm_dim1
        self.interm_dim2 = interm_dim2
        self.fc1 = nn.Linear(input_dim, interm_dim1)
        self.fc2 = nn.Linear(interm_dim1, interm_dim2)
        self.fc3 = nn.Linear(interm_dim2, 1)
        self.modules_sizes = [self.d * self.interm_dim1, self.interm_dim1, 
            self.interm_dim1 * self.interm_dim2, self.interm_dim2, self.
            interm_dim2, 1]
        self.w_opt = None
        self.alpha = None
        self.mixed_linead_weights = None

    def set_mixed_linear_weights(self, w):
        self.mixed_linead_weights = self.alpha * w + (1 - self.alpha
            ) * self.w_opt
        self.mixed_linead_weights.retain_grad()
        fc_parameters = torch.split(self.mixed_linead_weights, self.
            modules_sizes)
        ind = 0
        for module in self.modules():
            if type(module) == nn.Linear:
                module.weight = torch.nn.Parameter(fc_parameters[ind].view(
                    module.weight.shape))
                ind += 1
                module.bias = torch.nn.Parameter(fc_parameters[ind].view(
                    module.bias.shape))
                ind += 1

    def forward(self, x, w=None):
        if w is not None:
            assert w.requires_grad
            assert self.alpha is not None
            assert self.w_opt is not None
            self.set_mixed_linear_weights(w)
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'interm_dim1': 4, 'interm_dim2': 4}]
