from torch.nn import Module
import torch
from torch import Tensor
from torch.nn import MSELoss


def _split_masks_by_classes(pred: 'Tensor', target: 'Tensor') ->[]:
    """
    Split masks by classes

    Args:
        pred (Tensor): predicted masks of shape [B, C, H, W]
        target (Tensor): target masks of shape [B, C, H, W]

    Returns:
        List: list of masks pairs [pred, target], splitted by channels. List shape: [C, 2, B, H, W]
    """
    preds = torch.split(pred, 1, dim=1)
    targets = torch.split(target, 1, dim=1)
    return list(zip(preds, targets))


class Reduction:

    def __init__(self, method: 'str'='sum'):
        super().__init__()
        if method == 'sum':
            self._reduction = lambda x: x.sum(0)
            self._list_reduction = lambda x: sum(x)
        elif method == 'mean':
            self._reduction = lambda x: x.sum(0)
            self._list_reduction = lambda x: sum(x) / len(x)
        else:
            raise Exception(
                "Unexpected reduction '{}'. Possible values: [sum, mean]".
                format(method))

    def __call__(self, data):
        return self._reduction(data).unsqueeze(0)

    def reduct_list(self, data):
        return self._list_reduction(data).unsqueeze(0)


class MulticlassSegmentationLoss(Module):
    """
    Wrapper loss function to work with multiclass inference.
    This just split masks by classes and calculate :arg:`base_loss` for every class. After that all loss values summarized

    Args:
         base_loss (Module): basic loss object
    """

    def __init__(self, base_loss: 'Module', reduction: 'Reduction'=
        Reduction('sum')):
        super().__init__()
        self._base_loss = base_loss
        self._reduction = reduction

    def forward(self, output: 'Tensor', target: 'Tensor'):
        res = []
        for i, [p, t] in enumerate(_split_masks_by_classes(output, target)):
            res.append(self._base_loss(p, t))
        return self._reduction.reduct_list(res)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'base_loss': MSELoss()}]
