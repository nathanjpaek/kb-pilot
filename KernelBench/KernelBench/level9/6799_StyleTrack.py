import torch
import torch.nn as nn


def gram_matrix(input):
    """ gram matrix for feature assignments """
    a, b, c, d = input.size()
    allG = []
    for i in range(a):
        features = input[i].view(b, c * d)
        gram = torch.mm(features, features.t())
        gram = gram.div(c * d)
        allG.append(gram)
    return torch.stack(allG)


class StyleTrack(nn.Module):
    """ This modules tracks the content image style across many images
        for loading (partnered with next module). This is useful for
        e.g. maintaining color scheme of the content images
    """

    def forward(self, input):
        """ forward pass """
        gram = gram_matrix(input)
        self.value = gram
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
