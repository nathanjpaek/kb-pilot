import torch
import torch.nn as nn


class AveragePooling(nn.Module):

    def __init__(self):
        super(AveragePooling, self).__init__()
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
        x_now = x.clone()
        empty_mask = x_mask.eq(0).unsqueeze(2).expand_as(x_now)
        x_now.data.masked_fill_(empty_mask.data, 0)
        x_sum = torch.sum(x_now, 1)
        x_num = torch.sum(x_mask.eq(1).float(), 1).unsqueeze(1).expand_as(x_sum
            )
        x_num = torch.clamp(x_num, min=1)
        return x_sum / x_num


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
