import torch


class LinearBlock(torch.nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int') ->None:
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features, out_features)
        self.layer_2 = torch.nn.Linear(out_features, out_features)
        self.activation = torch.nn.LeakyReLU(0.01)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.dropout(self.activation(self.layer_1(x)))
        return self.activation(self.layer_2(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
