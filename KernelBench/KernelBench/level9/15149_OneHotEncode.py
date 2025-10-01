import torch
import torch.optim
import torch.nn as nn
import torch.nn.utils
import torch.autograd


def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    xview = input_tensor.view(-1, 1)
    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.
        device, dtype=torch.float)
    onehot.scatter_(1, xview, 1)
    return onehot.view(list(input_tensor.shape) + [-1])


class OneHotEncode(nn.Module):
    """ One-hot encoding layer """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return one_hot_encoding(x, self.num_classes)


def get_inputs():
    return [torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'num_classes': 4}]
