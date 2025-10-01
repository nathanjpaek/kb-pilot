import torch
import torch.nn as nn
import torch.optim
import torch.autograd
import torch.nn
import torch.nn.init


class FencepostModule(nn.Module):

    def __init__(self, input_dim, repr_dim, n_labels, disentangle=False,
        label_bias=True, span_bias=False, activation='tanh'):
        super(FencepostModule, self).__init__()
        self.disentangle = disentangle
        self.activation_name = activation
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()
        else:
            raise RuntimeError('Unknown activation function: %s' % self.
                activation_name)
        self.label_bias = label_bias
        self.span_bias = span_bias
        self.label_output_mlp = nn.Linear(input_dim, repr_dim, bias=True)
        self.label_output_projection = torch.nn.Parameter(data=torch.Tensor
            (repr_dim, n_labels))
        if self.span_bias:
            self.span_output_mlp = nn.Linear(input_dim, repr_dim, bias=True)
            self.span_output_projection = torch.nn.Parameter(data=torch.
                Tensor(repr_dim, 1))
        if self.label_bias:
            self.output_bias = torch.nn.Parameter(data=torch.Tensor(1,
                n_labels))
        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            if self.activation_name == 'tanh':
                torch.nn.init.xavier_uniform_(self.label_output_mlp.weight)
            else:
                torch.nn.init.kaiming_normal_(self.label_output_mlp.weight,
                    nonlinearity=self.activation_name)
            torch.nn.init.zeros_(self.label_output_mlp.bias)
            torch.nn.init.xavier_uniform_(self.label_output_projection)
            if self.span_bias:
                if self.activation_name == 'tanh':
                    torch.nn.init.xavier_uniform_(self.span_output_mlp.weight)
                else:
                    torch.nn.init.kaiming_uniform_(self.span_output_mlp.
                        weight, nonlinearity=self.activation_name)
                torch.nn.init.zeros_(self.span_output_mlp.bias)
                torch.nn.init.xavier_uniform_(self.span_output_projection)
            if self.label_bias:
                self.output_bias.fill_(0.0)
    """
        Input dimension is: (n_words + 2,  input_repr)
    """

    def forward(self, input):
        n_words = input.size()[0]
        input_repr = input.size()[1]
        if self.disentangle:
            input = torch.cat([input[:, 0::2], input[:, 1::2]], 1)
        fencepost_annotations = torch.cat([input[:-1, :input_repr // 2], -
            input[1:, input_repr // 2:]], 1)
        span_features = torch.unsqueeze(fencepost_annotations, 0
            ) - torch.unsqueeze(fencepost_annotations, 1)
        n_words = span_features.size()[0]
        span_features = span_features.view(n_words * n_words, -1)
        output = self.activation(self.label_output_mlp(span_features)
            ) @ self.label_output_projection
        if self.span_bias:
            output = output + self.activation(self.span_output_mlp(
                span_features)) @ self.span_output_projection
        if self.label_bias:
            output = output + self.output_bias
        output = output.view(n_words, n_words, -1)
        return output


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'repr_dim': 4, 'n_labels': 4}]
