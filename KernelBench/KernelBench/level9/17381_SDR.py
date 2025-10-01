import torch


class SDR(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.expr = 'bi,bi->b'

    def _batch_dot(self, x, y):
        return torch.einsum(self.expr, x, y)

    def forward(self, outputs, labels):
        if outputs.dtype != labels.dtype:
            outputs = outputs
        length = min(labels.shape[-1], outputs.shape[-1])
        labels = labels[..., :length].reshape(labels.shape[0], -1)
        outputs = outputs[..., :length].reshape(outputs.shape[0], -1)
        delta = 1e-07
        num = self._batch_dot(labels, labels)
        den = num + self._batch_dot(outputs, outputs) - 2 * self._batch_dot(
            outputs, labels)
        den = den.relu().add_(delta).log10()
        num = num.add_(delta).log10()
        return 10 * (num - den)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
