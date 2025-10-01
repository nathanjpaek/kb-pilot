import torch
import torch.nn as nn


class model(nn.Module):

    def __init__(self, input_shape=28 * 28, nr_classes=10):
        super(model, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape, 200)
        self.fc2 = nn.Linear(200, nr_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        log_softmax_out = nn.LogSoftmax()(x)
        return log_softmax_out

    def forward_inference(self, x):
        raise NotImplementedError

    def get_name(self):
        return 'default_model'


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
