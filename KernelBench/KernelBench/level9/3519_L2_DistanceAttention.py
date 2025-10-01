import torch
import torch.nn as nn
import torch.utils.data


class L2_DistanceAttention(nn.Module):

    def __init__(self):
        super(L2_DistanceAttention, self).__init__()

    def forward(self, input_hidden_traces, target_hidden_traces):
        standard_size = input_hidden_traces.size(0), input_hidden_traces.size(1
            ), input_hidden_traces.size(1)
        target_hidden_traces_square = (target_hidden_traces ** 2).sum(2
            ).unsqueeze(2).expand(standard_size)
        input_hidden_traces_square = (input_hidden_traces ** 2).transpose(1, 2
            ).sum(1).unsqueeze(1).expand(standard_size)
        input_target_mm = torch.bmm(target_hidden_traces,
            input_hidden_traces.transpose(1, 2))
        inner_distance = (target_hidden_traces_square +
            input_hidden_traces_square - 2 * input_target_mm)
        Attn = -inner_distance
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(standard_size)
        exp_Attn = torch.exp(Attn)
        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(standard_size)
        return Attn, inner_distance


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
