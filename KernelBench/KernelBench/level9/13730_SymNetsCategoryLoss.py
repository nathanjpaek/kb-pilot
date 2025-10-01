import torch
import torch.nn.functional as F


def split_half(x, dim):
    d = x.shape[dim] // 2
    return torch.split(x, d, dim=dim)


class ConcatSoftmax(torch.nn.Module):
    """
    Applies softmax to the concatenation of a list of tensors.
    """

    def __init__(self, dim: 'int'=1):
        """
        Arguments:
            dim: a dimension along which softmax will be computed
        """
        super().__init__()
        self.dim = dim

    def forward(self, *x: torch.Tensor):
        """
        Arguments:
            *x: A sequence of tensors to be concatenated
        """
        all_logits = torch.cat(x, dim=self.dim)
        return torch.nn.functional.softmax(all_logits, dim=self.dim)

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ['dim'])


class SymNetsCategoryLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax_fn = ConcatSoftmax()

    def forward(self, x, y, src_labels):
        x = self.softmax_fn(x, y)
        x, y = split_half(x, dim=1)
        x_loss = F.cross_entropy(x, src_labels)
        y_loss = F.cross_entropy(y, src_labels)
        return x_loss + y_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
