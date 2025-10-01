import torch
import torch.utils.data


class SoftmaxBCELoss(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bce = torch.nn.BCELoss(*args, **kwargs)

    def forward(self, output, target):
        probs = torch.nn.functional.softmax(output, dim=1)
        return self.bce(probs, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
