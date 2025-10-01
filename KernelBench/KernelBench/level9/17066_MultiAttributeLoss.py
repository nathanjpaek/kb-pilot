import torch
import torch.nn.functional as F


class MultiAttributeLoss(torch.nn.Module):

    def __init__(self):
        super(MultiAttributeLoss, self).__init__()

    def forward(self, input, target):
        product = 1
        count = len(input)
        for i in range(count):
            attribute_loss = F.cross_entropy(input[i], target[i])
            product *= attribute_loss
        geometric_mean = torch.pow(product, count)
        return geometric_mean


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
