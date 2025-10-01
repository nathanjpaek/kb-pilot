import math
import torch
import torch.optim
import torch.utils.data


class ChannelAttention(torch.nn.Module):

    def __init__(self, N_out, N_in, ratio=1):
        super(ChannelAttention, self).__init__()
        self.linear = torch.nn.functional.linear
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.N_in = N_in
        self.N_out = N_out
        self.weight_fc1 = torch.nn.Parameter(torch.Tensor(self.N_out, self.
            N_in // ratio, self.N_in))
        self.weight_fc2 = torch.nn.Parameter(torch.Tensor(self.N_out, self.
            N_in, self.N_in // ratio))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_fc1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight_fc2, a=math.sqrt(5))

    def forward(self, input):
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        avg_out = self._linear(torch.relu(self._linear(input_mean, self.
            weight_fc1)), self.weight_fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, self.
            weight_fc1)), self.weight_fc2)
        out = torch.sigmoid(avg_out + max_out)
        out = torch.reshape(out, [input.shape[0], self.N_out, input.shape[2
            ], self.N_in, 1, 1])
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-3)
        w_reshaped = w.reshape(1, w.shape[0], 1, w.shape[1], w.shape[2], 1)
        output = (in_reshaped * w_reshaped).sum(-2)
        return output


class ChannelAttentionGG(ChannelAttention):

    def __init__(self, N_h, N_out, N_h_in, N_in, ratio=1, bias=False):
        super(ChannelAttentionGG, self).__init__(N_out, N_in, ratio=ratio)
        self.N_h_in = N_h_in
        self.N_h = N_h
        self.weight_fc1 = torch.nn.Parameter(torch.rand(self.N_out, self.
            N_in // ratio, self.N_in, self.N_h_in))
        self.weight_fc2 = torch.nn.Parameter(torch.rand(self.N_out, self.
            N_in, self.N_in // ratio, self.N_h_in))
        self.reset_parameters()

    def forward(self, input):
        fc1, fc2 = self._left_action_of_h_grid()
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        avg_out = self._linear(torch.relu(self._linear(input_mean, fc1)), fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, fc1)), fc2)
        out = torch.sigmoid(avg_out + max_out)
        out = torch.reshape(out, [input.shape[0], self.N_out, self.N_h, -1,
            self.N_h_in, 1, 1])
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-4)
        w_reshaped = torch.reshape(w, [1, w.shape[0], w.shape[1], w.shape[2
            ], w.shape[3], w.shape[4], 1])
        output = (in_reshaped * w_reshaped).sum(-3)
        return output

    def _left_action_of_h_grid(self):
        fc1 = torch.stack([self.weight_fc1.roll(shifts=i, dims=-1) for i in
            range(self.N_h)], dim=1)
        fc2 = torch.stack([self.weight_fc2.roll(shifts=i, dims=-1) for i in
            range(self.N_h)], dim=1)
        return fc1, fc2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'N_h': 4, 'N_out': 4, 'N_h_in': 4, 'N_in': 4}]
