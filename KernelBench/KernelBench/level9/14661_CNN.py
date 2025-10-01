import torch


class CNN(torch.nn.Module):

    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv_1', torch.nn.Conv2d(1, 4, kernel_size=2))
        self.conv.add_module('dropout_1', torch.nn.Dropout())
        self.conv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('relu_1', torch.nn.ReLU())
        self.conv.add_module('conv_2', torch.nn.Conv2d(4, 8, kernel_size=2))
        self.conv.add_module('dropout_2', torch.nn.Dropout())
        self.conv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('relu_2', torch.nn.ReLU())
        self.fc = torch.nn.Sequential()
        self.fc.add_module('fc1', torch.nn.Linear(8, 32))
        self.fc.add_module('relu_3', torch.nn.ReLU())
        self.fc.add_module('dropout_3', torch.nn.Dropout())
        self.fc.add_module('fc2', torch.nn.Linear(32, n_classes))

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 8)
        return self.fc.forward(x)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'n_classes': 4}]
