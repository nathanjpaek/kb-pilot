import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def one_hot(val: 'torch.LongTensor', num: 'int', num_first: 'bool'=False
    ) ->torch.FloatTensor:
    """
    Overview:
        Convert a ``torch.LongTensor`` to one hot encoding.
        This implementation can be slightly faster than ``torch.nn.functional.one_hot``
    Arguments:
        - val (:obj:`torch.LongTensor`): each element contains the state to be encoded, the range should be [0, num-1]
        - num (:obj:`int`): number of states of the one hot encoding
        - num_first (:obj:`bool`): If ``num_first`` is False, the one hot encoding is added as the last; \\
            Otherwise as the first dimension.
    Returns:
        - one_hot (:obj:`torch.FloatTensor`)
    Example:
        >>> one_hot(2*torch.ones([2,2]).long(),3)
        tensor([[[0., 0., 1.],
                 [0., 0., 1.]],
                [[0., 0., 1.],
                 [0., 0., 1.]]])
        >>> one_hot(2*torch.ones([2,2]).long(),3,num_first=True)
        tensor([[[0., 0.], [1., 0.]],
                [[0., 1.], [0., 0.]],
                [[1., 0.], [0., 1.]]])
    """
    assert isinstance(val, torch.Tensor), type(val)
    assert val.dtype == torch.long
    assert len(val.shape) >= 1
    old_shape = val.shape
    val_reshape = val.reshape(-1, 1)
    ret = torch.zeros(val_reshape.shape[0], num, device=val.device)
    index_neg_one = torch.eq(val_reshape, -1).long()
    if index_neg_one.sum() != 0:
        val_reshape = torch.where(val_reshape != -1, val_reshape, torch.
            zeros(val_reshape.shape, device=val.device).long())
    try:
        ret.scatter_(1, val_reshape, 1)
        if index_neg_one.sum() != 0:
            ret = ret * (1 - index_neg_one)
    except RuntimeError:
        raise RuntimeError('value: {}\nnum: {}\t:val_shape: {}\n'.format(
            val_reshape, num, val_reshape.shape))
    if num_first:
        return ret.permute(1, 0).reshape(num, *old_shape)
    else:
        return ret.reshape(*old_shape, num)


class LabelSmoothCELoss(nn.Module):
    """
    Overview:
        Label smooth cross entropy loss.
    Interfaces:
        forward
    """

    def __init__(self, ratio: 'float') ->None:
        super().__init__()
        self.ratio = ratio

    def forward(self, logits: 'torch.Tensor', labels: 'torch.LongTensor'
        ) ->torch.Tensor:
        """
        Overview:
            Calculate label smooth cross entropy loss.
        Arguments:
            - logits (:obj:`torch.Tensor`): Predicted logits.
            - labels (:obj:`torch.LongTensor`): Ground truth.
        Returns:
            - loss (:obj:`torch.Tensor`): Calculated loss.
        """
        B, N = logits.shape
        val = float(self.ratio) / (N - 1)
        one_hot = torch.full_like(logits, val)
        one_hot.scatter_(1, labels.unsqueeze(1), 1 - val)
        logits = F.log_softmax(logits, dim=1)
        return -torch.sum(logits * one_hot.detach()) / B


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'ratio': 4}]
