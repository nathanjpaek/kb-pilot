import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationNet(nn.Module):

    def __init__(self, feature, hidden1, hidden2, output):
        """ Initialize a class NeuralNet.

        :param batch_size: int
        :param hidden: int
        """
        super(SegmentationNet, self).__init__()
        self.layer1 = nn.Linear(feature, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, output)

    def get_weight_norm(self):
        """ Return ||W||

        :return: float
        """
        layer_1_w_norm = torch.norm(self.layer1.weight, 2)
        layer_2_w_norm = torch.norm(self.layer2.weight, 2)
        layer_3_w_norm = torch.norm(self.layer3.weight, 2)
        return layer_1_w_norm + layer_2_w_norm + layer_3_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        out = inputs
        out = self.layer1(out)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.layer3(out)
        out = F.softmax(out, dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature': 4, 'hidden1': 4, 'hidden2': 4, 'output': 4}]
