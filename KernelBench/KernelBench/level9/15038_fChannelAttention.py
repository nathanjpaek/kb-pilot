import math
import torch
import torch.optim
import torch.utils.data


class fChannelAttention(torch.nn.Module):

    def __init__(self, N_in, ratio=1):
        super(fChannelAttention, self).__init__()
        self.N_in = N_in
        self.ratio = ratio
        self.weight_fc1 = torch.nn.Parameter(torch.Tensor(self.N_in //
            ratio, self.N_in))
        self.weight_fc2 = torch.nn.Parameter(torch.Tensor(self.N_in, self.
            N_in // ratio))
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
        out = torch.reshape(out, [input.shape[0], self.N_in, 1, 1])
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-3)
        w_reshaped = w.reshape(1, w.shape[0], w.shape[1], 1)
        output = (in_reshaped * w_reshaped).sum(-2)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'N_in': 4}]
