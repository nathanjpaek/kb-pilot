import torch
import torch.utils.data


class DiscreteCrossEntropyLoss(torch.nn.Module):

    def __init__(self, in_features, num_classes):
        super(DiscreteCrossEntropyLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(in_features, num_classes)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target, mask=None):
        x = self.fc(x)
        loss = self.cross_entropy_loss(x, target)
        if mask is not None:
            loss = loss * mask
        return loss

    def pack_init_args(self):
        args = {'in_features': self.in_features, 'num_classes': self.
            num_classes}
        return args


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'num_classes': 4}]
