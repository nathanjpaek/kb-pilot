import torch
import torch.nn as nn


class RadialPredictionLayer(torch.nn.Module):
    """ The RPL classification layer with fixed prototypes
    """

    def __init__(self, in_features, out_features):
        super(RadialPredictionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prototypes = nn.Parameter(torch.diag(torch.ones(self.
            in_features)), requires_grad=False)

    def forward(self, x):
        return -((x[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(dim=2
            ).sqrt()

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features,
            self.out_features)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
