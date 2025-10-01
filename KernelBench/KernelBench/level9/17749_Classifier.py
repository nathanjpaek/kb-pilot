import torch
import torch.nn as nn
import torch.utils.data


class Classifier(nn.Module):

    def __init__(self, feature_dim, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(int(feature_dim * 2), classes)

    def forward(self, di_z, ds_z):
        z = torch.cat((di_z, ds_z), dim=1)
        y = self.classifier(z)
        return y


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4, 'classes': 4}]
