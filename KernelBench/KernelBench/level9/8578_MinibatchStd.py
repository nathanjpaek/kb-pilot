import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch.nn


class MinibatchStd(nn.Module):
    """
    Adds the aveage std of each data point over a
    slice of the minibatch to that slice as a new
    feature map. This gives an output with one extra
    channel.
    Arguments:
        group_size (int): Number of entries in each slice
            of the batch. If <= 0, the entire batch is used.
            Default value is 4.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """

    def __init__(self, group_size=4, eps=1e-08, *args, **kwargs):
        super(MinibatchStd, self).__init__()
        if group_size is None or group_size <= 0:
            group_size = 0
        assert group_size != 1, 'Can not use 1 as minibatch std group size.'
        self.group_size = group_size
        self.eps = eps

    def forward(self, input, **kwargs):
        """
        Add a new feature map to the input containing the average
        standard deviation for each slice.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        group_size = self.group_size or input.size(0)
        assert input.size(0
            ) >= group_size, 'Can not use a smaller batch size ' + '({}) than the specified '.format(
            input.size(0)) + 'group size ({}) '.format(group_size
            ) + 'of this minibatch std layer.'
        assert input.size(0
            ) % group_size == 0, 'Can not use a batch of a size ' + '({}) that is not '.format(
            input.size(0)) + 'evenly divisible by the group size ({})'.format(
            group_size)
        x = input
        y = input.view(group_size, -1, *input.size()[1:])
        y = y.float()
        y -= y.mean(dim=0, keepdim=True)
        y = torch.mean(y ** 2, dim=0)
        y = torch.sqrt(y + self.eps)
        y = torch.mean(y.view(y.size(0), -1), dim=-1)
        y = y.view(-1, *([1] * (input.dim() - 1)))
        y = y
        y = y.repeat(group_size, *([1] * (y.dim() - 1)))
        y = y.expand(y.size(0), 1, *x.size()[2:])
        x = torch.cat([x, y], dim=1)
        return x

    def extra_repr(self):
        return 'group_size={}'.format(self.group_size or '-1')


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
