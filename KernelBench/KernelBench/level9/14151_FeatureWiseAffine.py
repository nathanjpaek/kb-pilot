import torch


class BaseModule(torch.nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FeatureWiseAffine(BaseModule):

    def __init__(self):
        super(FeatureWiseAffine, self).__init__()

    def forward(self, x, scale, shift):
        outputs = scale * x + shift
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
