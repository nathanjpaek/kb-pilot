import torch


class NeuralNet(torch.nn.Module):

    def __init__(self, input_features, hidden_layer_size, output_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_features, hidden_layer_size)
        self.l2 = torch.nn.Linear(hidden_layer_size, output_classes)

    def forward(self, X):
        hidden_layer = torch.sigmoid(self.l1(X))
        return self.l2(hidden_layer)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_features': 4, 'hidden_layer_size': 1,
        'output_classes': 4}]
