import torch
import torch.nn as nn


class Biaffine(nn.Module):

    def __init__(self, in_features, out_features=1, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in_features + bias[0]
        self.linear_output_size = out_features * (in_features + bias[1])
        self.linear = nn.Linear(in_features=self.linear_input_size,
            out_features=self.linear_output_size, bias=False)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, _dim1 = input1.size()
        batch_size, len2, _dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new_ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=-1)
        if self.bias[1]:
            ones = input2.data.new_ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=-1)
        affine = self.linear(input1)
        affine = affine.reshape(batch_size, len1 * self.out_features, -1)
        biaffine = torch.bmm(affine, input2.transpose(1, 2)).transpose(1, 2
            ).contiguous()
        biaffine = biaffine.reshape((batch_size, len2, len1, -1)).squeeze(-1)
        return biaffine


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
