import torch
import torch.utils.data


class ZonoDenseLayer(torch.nn.Module):
    """
    Class implementing a dense layer on a zonotope.
    Bias is only added to the zeroth term.
    """

    def __init__(self, in_features: 'int', out_features: 'int'):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros(
            out_features, in_features), std=torch.ones(out_features,
            in_features)))
        self.bias = torch.nn.Parameter(torch.normal(mean=torch.zeros(
            out_features), std=torch.ones(out_features)))

    def __call__(self, x: "'torch.Tensor'") ->'torch.Tensor':
        return self.forward(x)

    def forward(self, x: "'torch.Tensor'") ->'torch.Tensor':
        """
        Forward pass through the dense layer.

        :param x: input zonotope to the dense layer.
        :return: zonotope after being pushed through the dense layer.
        """
        x = self.zonotope_matmul(x)
        x = self.zonotope_add(x)
        return x

    def zonotope_matmul(self, x: "'torch.Tensor'") ->'torch.Tensor':
        """
        Matrix multiplication for dense layer.

        :param x: input to the dense layer.
        :return: zonotope after weight multiplication.
        """
        return torch.matmul(x, torch.transpose(self.weight, 0, 1))

    def zonotope_add(self, x: "'torch.Tensor'") ->'torch.Tensor':
        """
        Modification required compared to the normal torch dense layer.
        The bias is added only to the central zonotope term and not the error terms.

        :param x: zonotope input to have the bias added.
        :return: zonotope with the bias added to the central (first) term.
        """
        x[0] = x[0] + self.bias
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
