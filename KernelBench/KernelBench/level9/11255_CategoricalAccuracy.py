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


class CategoricalAccuracy(Accuracy):

    def __init__(self):
        super().__init__()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'):
        """
        :param input: [B, T, V]
        :param target: [B, T]
        :return:
        """
        input = input.softmax(-1)
        categorical_input = input.argmax(-1)
        return super().forward(categorical_input, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
