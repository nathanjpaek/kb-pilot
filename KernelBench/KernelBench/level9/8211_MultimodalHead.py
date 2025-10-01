import torch
from torch import nn


class MultimodalHead(nn.Module):
    """
    Multimodal head for the conv net outputs.
    This layer concatenate the outputs of audio and visual convoluational nets
    and performs a fully-connected projection
    """

    def __init__(self, dim_in, num_classes, dropout_rate=0.0, act_func=
        'softmax'):
        """
        Args:
            dim_in (int): the channel dimensions of the visual/audio inputs.
            num_classes (int): the channel dimension of the output.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(MultimodalHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        if act_func == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                '{} is not supported as an activationfunction.'.format(
                act_func))

    def forward(self, x, y):
        xy_cat = torch.cat((x, y), dim=-1)
        if hasattr(self, 'dropout'):
            xy_cat = self.dropout(xy_cat)
        xy_cat = self.projection(xy_cat)
        if not self.training:
            xy_cat = self.act(xy_cat)
        return xy_cat


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': [4, 4], 'num_classes': 4}]
