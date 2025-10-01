import torch
import torch.nn as nn


class LinearWeightNorm(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearWeightNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        self.linear = nn.utils.weight_norm(self.linear)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.
            in_features, self.out_features, self.bias is not None)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            out = self(x).view(-1, self.linear.out_features)
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-06)
            self.linear.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.linear.bias is not None:
                self.linear.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.linear(input)


class NICEMLPBlock(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, activation):
        super(NICEMLPBlock, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, hidden_features, bias=True)
        self.fc3 = LinearWeightNorm(hidden_features, out_features, bias=True)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        return out

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            out = self.activation(self.fc1(x))
            out = self.activation(self.fc2(out))
            out = self.fc3.init(out, init_scale=0.0 * init_scale)
            return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'hidden_features': 4,
        'activation': 'relu'}]
