import torch
import torch.nn as nn


def timestep_dropout(inputs, p=0.5, batch_first=True):
    """
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    """
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    batch_size, _time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1 - p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask


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


class NonlinearMLP(nn.Module):

    def __init__(self, in_feature, out_feature, activation=None, bias=True):
        super(NonlinearMLP, self).__init__()
        if activation is None:
            self.activation = lambda x: x
        else:
            assert callable(activation)
            self.activation = activation
        self.bias = bias
        self.linear = nn.Linear(in_features=in_feature, out_features=
            out_feature, bias=bias)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return self.activation(linear_out)


class BiaffineScorer(nn.Module):

    def __init__(self, input_size, ffnn_size, num_cls, ffnn_drop=0.33):
        super(BiaffineScorer, self).__init__()
        self.ffnn_size = ffnn_size
        self.ffnn_drop = ffnn_drop
        self._act = nn.ELU()
        self.mlp_start = NonlinearMLP(in_feature=input_size, out_feature=
            ffnn_size, activation=self._act)
        self.mlp_end = NonlinearMLP(in_feature=input_size, out_feature=
            ffnn_size, activation=self._act)
        self.span_biaff = Biaffine(ffnn_size, num_cls, bias=(True, True))

    def forward(self, enc_hn):
        start_feat = self.mlp_start(enc_hn)
        end_feat = self.mlp_end(enc_hn)
        if self.training:
            start_feat = timestep_dropout(start_feat, self.ffnn_drop)
            end_feat = timestep_dropout(end_feat, self.ffnn_drop)
        span_score = self.span_biaff(start_feat, end_feat)
        return span_score


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'ffnn_size': 4, 'num_cls': 4}]
