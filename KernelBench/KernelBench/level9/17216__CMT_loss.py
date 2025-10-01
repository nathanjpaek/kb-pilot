import torch
import torch.nn as nn


class _CMT_loss(nn.Module):

    def __init__(self):
        super(_CMT_loss, self).__init__()
        self.d = nn.PairwiseDistance()

    def forward(self, feat, sematics):
        """
        :param feat: features of images or images. bs * d. d is the length of word vector.
        :param sematics: sematics of sketches. bs * d. d is the length of word vector.
        :return: loss
        """
        return torch.mean(self.d(feat.float(), sematics.float()) ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
