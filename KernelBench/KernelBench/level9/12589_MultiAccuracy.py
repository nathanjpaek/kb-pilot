import torch


class MultiAccuracy(torch.nn.Module):
    """Calculates accuracy for multiclass inputs (batchsize, feature length) by determining the most likely class
    using argmax -> (batchsize,) and then comparing with targets which are also (batchsize,)
    """

    def __init__(self):
        super(MultiAccuracy, self).__init__()

    def forward(self, outputs, targets):
        if outputs.shape != targets.shape:
            outputs = torch.argmax(outputs, dim=-1)
        return torch.sum(outputs == targets, dim=-1) / targets.shape[-1]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
