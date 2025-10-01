import torch
import torch.nn


class LayerCake(torch.nn.Module):

    def __init__(self, D_in, H1, H2, H3, H4, H5, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LayerCake, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, H4)
        self.linear5 = torch.nn.Linear(H4, H5)
        self.linear6 = torch.nn.Linear(H5, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        x = self.linear1(x).clamp(min=0)
        x = self.linear2(x).clamp(min=0)
        x = self.linear3(x).clamp(min=0)
        x = self.linear4(x).clamp(min=0)
        x = self.linear5(x).clamp(min=0)
        y_pred = self.linear6(x)
        return y_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'H1': 4, 'H2': 4, 'H3': 4, 'H4': 4, 'H5': 4,
        'D_out': 4}]
