import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms.functional as F
from torch.nn import functional as F


class MetricCELoss(nn.Module):
    """ Cross-entropy loss for metric learning with a specified feature size.
    In addition, there exists a ReLU layer to pre-process the input feature.

    Args:
        feature_size (int): usually 128, 256, 512 ...
        num_classes (int): num of classes when training
    """

    def __init__(self, feature_size, num_classes):
        super(MetricCELoss, self).__init__()
        self.in_features = feature_size
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features,
            self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        """ Output the logit.

        Args:
            feature (:obj:`torch.FloatTensor`): image features from previous layers.

        Returns:
            :obj:`torch.FloatTensor`: logit
        """
        output = F.linear(F.relu(feature), self.weight)
        return output

    def forward(self, feature, label):
        """ Calculate MetricCE loss.

        Args:
            feature (:obj:`torch.FloatTensor`): image features from previous layers.
            label (:obj:`torch.LongTensor`): image labels.

        Returns:
            (:obj:`torch.FloatTensor`, :obj:`torch.FloatTensor`):

            * loss: MetricCE loss over the batch.
            * logit: logit output after ReLU.
        """
        output = self.output(feature)
        loss = self.ce(output, label)
        return loss, output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'num_classes': 4}]
