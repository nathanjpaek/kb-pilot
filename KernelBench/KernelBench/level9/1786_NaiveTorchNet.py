import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd
import torch.optim as optim


class NaiveTorchNet(nn.Module):
    """A reimplementation of from-scratch NaiveNet using PyTorch"""

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate=0.1
        ):
        super().__init__()
        self.hidden = nn.Linear(input_nodes, hidden_nodes, bias=False)
        self.output = nn.Linear(hidden_nodes, output_nodes, bias=False)
        self.lr = learn_rate
        self.activation_function = nn.Sigmoid()
        self.optimizer = optim.SGD(self.parameters(), lr=learn_rate)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        """Overrides the built in"""
        x = self.activation_function(self.hidden(x))
        x = self.activation_function(self.output(x))
        return x

    def query(self, inputs):
        """Takes an input to the net and returns an output via forward computation"""
        if type(inputs) != torch.autograd.variable.Variable:
            inputs = Variable(torch.Tensor(inputs))
        return {'i': inputs, 'fo': self.forward(inputs)}

    def learn(self, targets, input_layers):
        if type(targets) != torch.autograd.variable.Variable:
            targets = Variable(torch.Tensor(targets))
        final_outputs = input_layers['fo']
        output_errors = self.loss_function(final_outputs, targets)
        self.optimizer.zero_grad()
        output_errors.backward()
        self.optimizer.step()
        return output_errors, final_outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nodes': 4, 'hidden_nodes': 4, 'output_nodes': 4}]
