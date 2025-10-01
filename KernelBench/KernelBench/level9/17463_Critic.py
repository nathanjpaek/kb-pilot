import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        self.l7 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l8 = nn.Linear(hidden_dim, hidden_dim)
        self.l9 = nn.Linear(hidden_dim, 1)
        self.l10 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l11 = nn.Linear(hidden_dim, hidden_dim)
        self.l12 = nn.Linear(hidden_dim, 1)
        self.l13 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l14 = nn.Linear(hidden_dim, hidden_dim)
        self.l15 = nn.Linear(hidden_dim, 1)
        self.l16 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l17 = nn.Linear(hidden_dim, hidden_dim)
        self.l18 = nn.Linear(hidden_dim, 1)
        self.l19 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l20 = nn.Linear(hidden_dim, hidden_dim)
        self.l21 = nn.Linear(hidden_dim, 1)
        self.l22 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l23 = nn.Linear(hidden_dim, hidden_dim)
        self.l24 = nn.Linear(hidden_dim, 1)
        self.l25 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l26 = nn.Linear(hidden_dim, hidden_dim)
        self.l27 = nn.Linear(hidden_dim, 1)
        self.l28 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l29 = nn.Linear(hidden_dim, hidden_dim)
        self.l30 = nn.Linear(hidden_dim, 1)
        self.apply(weight_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        q3 = F.relu(self.l7(sa))
        q3 = F.relu(self.l8(q3))
        q3 = self.l9(q3)
        q4 = F.relu(self.l10(sa))
        q4 = F.relu(self.l11(q4))
        q4 = self.l12(q4)
        q5 = F.relu(self.l13(sa))
        q5 = F.relu(self.l14(q5))
        q5 = self.l15(q5)
        q6 = F.relu(self.l16(sa))
        q6 = F.relu(self.l17(q6))
        q6 = self.l18(q6)
        q7 = F.relu(self.l19(sa))
        q7 = F.relu(self.l20(q7))
        q7 = self.l21(q7)
        q8 = F.relu(self.l22(sa))
        q8 = F.relu(self.l23(q8))
        q8 = self.l24(q8)
        q9 = F.relu(self.l25(sa))
        q9 = F.relu(self.l26(q9))
        q9 = self.l27(q9)
        q10 = F.relu(self.l28(sa))
        q10 = F.relu(self.l29(q10))
        q10 = self.l30(q10)
        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10

    def Qvalue(self, state, action, head=1):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        q3 = F.relu(self.l7(sa))
        q3 = F.relu(self.l8(q3))
        q3 = self.l9(q3)
        q4 = F.relu(self.l10(sa))
        q4 = F.relu(self.l11(q4))
        q4 = self.l12(q4)
        q5 = F.relu(self.l13(sa))
        q5 = F.relu(self.l14(q5))
        q5 = self.l15(q5)
        q6 = F.relu(self.l16(sa))
        q6 = F.relu(self.l17(q6))
        q6 = self.l18(q6)
        q7 = F.relu(self.l19(sa))
        q7 = F.relu(self.l20(q7))
        q7 = self.l21(q7)
        q8 = F.relu(self.l22(sa))
        q8 = F.relu(self.l23(q8))
        q8 = self.l24(q8)
        q9 = F.relu(self.l25(sa))
        q9 = F.relu(self.l26(q9))
        q9 = self.l27(q9)
        q10 = F.relu(self.l28(sa))
        q10 = F.relu(self.l29(q10))
        q10 = self.l30(q10)
        q_dict = {(1): q1, (2): q2, (3): q3, (4): q4, (5): q5, (6): q6, (7):
            q7, (8): q8, (9): q9, (10): q10}
        if head < 10:
            return q_dict[head], q_dict[head + 1]
        else:
            return q_dict[10], q_dict[1]


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
