import torch


class _Metric(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'):
        raise NotImplementedError()


class Accuracy(_Metric):

    def __init__(self):
        super().__init__()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'):
        """
        :param input: [B, L]
        :param target: [B, L]
        :return:
        """
        bool_acc = input.long() == target.long()
        return bool_acc.sum() / bool_acc.numel()


class MockAccuracy(Accuracy):

    def __init__(self):
        super().__init__()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'):
        return super().forward(input, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
