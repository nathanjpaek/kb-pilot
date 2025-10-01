import torch
import torch.nn as nn


class MaxPooling(nn.Module):

    def __init__(self):
        super(MaxPooling, self).__init__()
        self.MIN = -1000000.0
    """
     (item, subitem) can be (word, characters), or (sentence, words)
     x: num_items x max_subitem_size x input_size
     x_mask: num_items x max_subitem_size
     return num_items x input_size
    """

    def forward(self, x, x_mask):
        """
         x_output: num_items x input_size x 1 --> num_items x input_size
        """
        empty_mask = x_mask.eq(0).unsqueeze(2).expand_as(x)
        x_now = x.clone()
        x_now.data.masked_fill_(empty_mask.data, self.MIN)
        x_output = x_now.max(1)[0]
        x_output.data.masked_fill_(x_output.data.eq(self.MIN), 0)
        return x_output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
