import torch
import torch.nn as nn
import torch.nn.functional as F


class MySoftmax(nn.Module):

    def forward(self, input_):
        batch_size = input_.size()[0]
        output_ = torch.stack([F.softmax(input_[i]) for i in range(
            batch_size)], 0)
        return output_


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
