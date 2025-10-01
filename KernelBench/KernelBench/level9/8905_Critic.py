import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, hidden_size, action, num_inputs, spp_num_outputs,
        data_width=8):
        super(Critic, self).__init__()
        self.action = action
        self.num_outputs = self.action.shape[0]
        self.num_inputs = num_inputs
        self.spp_data_width = data_width
        self.spp_num_outputs = spp_num_outputs
        self.fc_num_inputs = sum([(i * i) for i in self.spp_num_outputs])
        """
        self.Cbx_res_block1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.RReLU(0.66, 0.99),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(0.33, 0.66),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )

        self.Cbx_res_block2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.RReLU(0.66, 0.99),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(0.33, 0.66),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )
        """
        self.cfc11 = nn.Linear(self.num_outputs, hidden_size)
        self.ln11 = nn.LayerNorm(hidden_size)
        self.relu11 = nn.RReLU(0.01, 0.33)
        self.cfc12 = nn.Linear(self.num_inputs, hidden_size)
        self.ln12 = nn.LayerNorm(hidden_size)
        self.relu12 = nn.RReLU(0.01, 0.33)
        self.cfc2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size * 2)
        self.relu2 = nn.RReLU(0.33, 0.66)
        self.cfc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.relu3 = nn.RReLU(0.66, 0.99)
        self.cfc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln4 = nn.LayerNorm(hidden_size // 2)
        self.drop = nn.Dropout()
        self.V = nn.Linear(hidden_size // 2, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, states, actions):
        states = states.squeeze()
        actions = actions.squeeze()
        x_b = actions
        x_b = x_b.reshape(-1, self.num_outputs)
        x_b = self.cfc11(x_b)
        x_b = self.ln11(x_b)
        x_b = self.relu11(x_b)
        x_Cbx = states.squeeze()
        x_Cbx = self.cfc12(x_Cbx)
        x_Cbx = self.ln12(x_Cbx)
        x_Cbx = self.relu12(x_Cbx)
        x = torch.cat([x_b, x_Cbx], -1)
        x = self.cfc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.cfc3(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.cfc4(x)
        x = self.ln4(x)
        x = self.drop(x)
        V = self.V(x)
        return V


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'action': torch.rand([4, 4]),
        'num_inputs': 4, 'spp_num_outputs': [4, 4]}]
