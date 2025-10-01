import torch
import torch.nn as nn


class StateAttention(nn.Module):

    def __init__(self):
        super(StateAttention, self).__init__()
        self.sm = nn.Softmax(dim=1)

    def forward(self, a_t, r_t, input_embedding, padded_mask):
        new_a_t = torch.zeros_like(a_t)
        for i in range(a_t.shape[1]):
            if i == 0:
                new_a_t[:, i] = a_t[:, 0] * r_t[:, 0]
            else:
                window = a_t[:, i - 1:i + 1]
                window_sum = window[:, 0] * r_t[:, 1] + window[:, 1] * r_t[:, 0
                    ]
                new_a_t[:, i - 1] += (1 - padded_mask[:, i]) * window_sum
                new_a_t[:, i] += padded_mask[:, i] * window_sum
        new_a_t = new_a_t.unsqueeze(dim=1)
        output = torch.matmul(new_a_t, input_embedding).squeeze(dim=1)
        return output, new_a_t.squeeze(dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
