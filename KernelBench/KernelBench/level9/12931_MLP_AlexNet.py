import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_AlexNet(nn.Module):
    """ The last fully connected part of LeNet MNIST:
    https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    """

    def __init__(self, input_nc, input_width, input_height, dropout_prob=
        0.0, **kwargs):
        super(MLP_AlexNet, self).__init__()
        self.dropout_prob = dropout_prob
        ngf = input_nc * input_width * input_height
        self.fc1 = nn.Linear(ngf, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        return F.log_softmax(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nc': 4, 'input_width': 4, 'input_height': 4}]
