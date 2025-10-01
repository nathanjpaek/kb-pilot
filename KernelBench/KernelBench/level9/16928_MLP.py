import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


def create_all_possible_moves(m, n):
    """Create all moves on a (m,n) board."""
    moves = []
    for i in range(m):
        for j in range(n):
            moves.append((i, j))
    return list(set(moves))


class MLP(nn.Module):
    """3-layer MLP for AlphaZero"""

    def __init__(self, board_size=15, num_hidden1=2000, num_hidden2=1000):
        super(MLP, self).__init__()
        self.board_size = board_size
        self.all_moves = create_all_possible_moves(board_size, board_size)
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        if len(self.all_moves) != self.board_size ** 2:
            raise ValueError("moves and board don't match")
        self.fc1 = nn.Linear(self.board_size ** 2, self.num_hidden1)
        self.fc2 = nn.Linear(self.num_hidden1, self.num_hidden2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc3 = nn.Linear(self.num_hidden2, self.board_size ** 2)
        self.fc4 = nn.Linear(self.num_hidden2, 1)

    def forward(self, x):
        x = x.view(-1, self.board_size ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        p = F.relu(self.fc3(x))
        p = self.logsoftmax(p).exp()
        v = torch.tanh(self.fc4(x))
        return p, v


def get_inputs():
    return [torch.rand([4, 225])]


def get_init_inputs():
    return [[], {}]
