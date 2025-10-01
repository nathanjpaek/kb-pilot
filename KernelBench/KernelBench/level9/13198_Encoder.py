import torch
import torch.nn as nn
import torch.nn
import torch.nn.init
import torch.optim


class Model(nn.Module):
    """ Class representing sampleable neural network model """

    def num_params(self):
        """ Get the number of model parameters. """
        return sum(p.numel() for p in self.parameters())

    def summary(self, hashsummary=False):
        None
        None
        self.num_params()
        None
        None
        if hashsummary:
            None
            for idx, hashvalue in enumerate(self.hashsummary()):
                None

    def hashsummary(self):
        """ Print a model summary - checksums of each layer parameters """
        children = list(self.children())
        result = []
        for child in children:
            result.extend(hashlib.sha256(x.detach().cpu().numpy().tobytes()
                ).hexdigest() for x in child.parameters())
        return result

    def loss(self, x_data, y_true, reduce='mean'):
        """ Forward propagate network and return a value of loss function """
        if reduce not in (None, 'sum', 'mean'):
            raise ValueError('`reduce` must be either None, `sum`, or `mean`!')
        y_pred = self(x_data)
        return y_pred, self.loss_value(x_data, y_true, y_pred, reduce=reduce)

    def loss_value(self, x_data, y_true, y_pred, reduce=None):
        """ Calculate a value of loss function """
        raise NotImplementedError


class Encoder(Model):
    """ Linear encoder """

    def __init__(self, c_in, c_out, affine=True):
        super(Encoder, self).__init__()
        assert c_out % 2 == 0
        self.fc1 = nn.Linear(c_in, c_in // 2)
        self.fc2 = nn.Linear(c_in // 2, c_in)

    def forward(self, x):
        x = torch.relu(x)
        x = self.fc1(x)
        return self.fc2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c_in': 4, 'c_out': 4}]
