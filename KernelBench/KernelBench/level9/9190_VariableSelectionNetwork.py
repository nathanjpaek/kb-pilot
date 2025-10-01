import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """**The unit of gating operation that maps the input to the range of 0-1 and multiple original input through the
    sigmoid function.**

    """

    def __init__(self, input_size, hidden_layer_size, dropout_rate,
        activation=None):
        """

        :param input_size: Number of features
        :param hidden_layer_size: The size of nn.Linear layer, global default is 160
        :param dropout_rate: The rate of linear layer parameters randomly discarded during training
        :param activation: activation function used to activate raw input, default is None
        """
        super(GatedLinearUnit, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        self.W4 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        if self.activation_name:
            self.activation = getattr(nn, self.activation_name)()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        if self.dropout_rate:
            x = self.dropout(x)
        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)
        return output


class GateAddNormNetwork(nn.Module):
    """**Units that adding gating output to skip connection improves generalization.**"""

    def __init__(self, input_size, hidden_layer_size, dropout_rate,
        activation=None):
        """

        :param input_size: Number of features
        :param hidden_layer_size: The size of nn.Linear layer, global default is 160
        :param dropout_rate: The rate of linear layer parameters randomly discarded during training
        :param activation: activation function used to activate raw input, default is None
        """
        super(GateAddNormNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.GLU = GatedLinearUnit(self.input_size, self.hidden_layer_size,
            self.dropout_rate, activation=self.activation_name)
        self.LayerNorm = nn.LayerNorm(self.hidden_layer_size)

    def forward(self, x, skip):
        output = self.LayerNorm(self.GLU(x) + skip)
        return output


class GatedResidualNetwork(nn.Module):
    """**GRN main module, which divides all inputs into two ways, calculates the gating one way for linear mapping twice and
    passes the original input to GateAddNormNetwork together. ** """

    def __init__(self, hidden_layer_size, input_size=None, output_size=None,
        dropout_rate=None):
        """

        :param hidden_layer_size: The size of nn.Linear layer, global default is 160
        :param input_size: Number of features
        :param output_size: Number of features
        :param dropout_rate: The rate of linear layer parameters randomly discarded during training
        """
        super(GatedResidualNetwork, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size if input_size else self.hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.W1 = torch.nn.Linear(self.hidden_layer_size, self.
            hidden_layer_size)
        self.W2 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        if self.output_size:
            self.skip_linear = torch.nn.Linear(self.input_size, self.
                output_size)
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                self.output_size, self.dropout_rate)
        else:
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                self.hidden_layer_size, self.dropout_rate)
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ('W2' in name or 'W3' in name) and 'bias' not in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in',
                    nonlinearity='leaky_relu')
            elif ('skip_linear' in name or 'W1' in name
                ) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        n2 = F.elu(self.W2(x))
        n1 = self.W1(n2)
        if self.output_size:
            output = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            output = self.glu_add_norm(n1, x)
        return output


class VariableSelectionNetwork(nn.Module):
    """**Feature selection module, which inputs a vector stitched into all features, takes the weights of each
    feature and multiply with the original input as output. ** """

    def __init__(self, hidden_layer_size, dropout_rate, output_size, input_size
        ):
        """

        :param hidden_layer_size: The size of nn.Linear layer, global default is 160
        :param dropout_rate: The rate of linear layer parameters randomly discarded during training
        :param output_size: Number of features
        :param input_size: Number of features
        """
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.flattened_grn = GatedResidualNetwork(self.hidden_layer_size,
            input_size=self.input_size, output_size=self.output_size,
            dropout_rate=self.dropout_rate)

    def forward(self, x):
        embedding = x
        flatten = torch.flatten(embedding, start_dim=1)
        mlp_outputs = self.flattened_grn(flatten)
        sparse_weights = F.softmax(mlp_outputs, dim=-1).mean(-2)
        combined = sparse_weights * flatten
        return combined, sparse_weights


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_layer_size': 1, 'dropout_rate': 0.5, 'output_size':
        4, 'input_size': 4}]
