import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):

    def __init__(self, board_width, board_height):
        super(LinearNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.model = nn.Linear(in_features=4 * self.board_width * self.
            board_height, out_features=self.board_width * self.board_height + 1
            )

    def forward(self, state_input):
        B = state_input.shape[0]
        x = state_input.reshape(B, -1)
        x = self.model(x)
        x_act, x_val = x[:, :-2], x[:, -1]
        x_act = F.sigmoid(x_act)
        return x_act, x_val


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'board_width': 4, 'board_height': 4}]
