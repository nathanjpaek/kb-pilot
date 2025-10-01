import torch
from torch import nn


class DenseCrossEntropy(nn.Module):
    """ The CrossEntropy loss that takes the one-hot
    vector of the gt label as the input, should be equivalent to the 
    standard CrossEntropy implementation. The one-hot vector
    is meant for the ArcFaceLoss and CutMix augmentation

    Args:
        x: the output of the model.
        target: the one-hot ground-truth label
    """

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
