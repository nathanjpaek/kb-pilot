import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, hidden_size, action, num_inputs, num_output,
        spp_num_outputs=[1, 2, 4], data_width=8):
        super(Actor, self).__init__()
        self.action = action
        self.num_inputs = num_inputs
        self.num_outputs = num_output
        self.spp_data_width = data_width
        self.spp_num_outputs = spp_num_outputs
        self.fc_num_inputs = sum([(i * i) for i in self.spp_num_outputs])
        """
        #设计残差模块用于Cb-x的向量输入
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
        self.afc1 = nn.Linear(self.num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.RReLU(0.01, 0.33)
        self.afc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.RReLU(0.33, 0.66)
        self.afc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.relu3 = nn.RReLU(0.66, 0.99)
        self.afc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln4 = nn.LayerNorm(hidden_size // 2)
        self.drop = nn.Dropout()
        self.mu = nn.Linear(hidden_size // 2, self.num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, states):
        res_Cbx = states
        """
        res_Cbx = res_Cbx.unsqueeze(2)
        res_Cbx = res_Cbx.unsqueeze(3)
        res_Cbx = res_Cbx.view(res_Cbx.size(0), 1, self.spp_data_width, res_Cbx.size(1) // self.spp_data_width)

        x_Cbx = self.Cbx_res_block1(res_Cbx)
        x_Cbx += res_Cbx
        res_Cbx = x_Cbx
        x_Cbx = self.Cbx_res_block2(res_Cbx)
        x_Cbx += res_Cbx

        x_Cbx = spatial_pyramid_pool(x_Cbx, x_Cbx.size(0), [x_Cbx.size(2), x_Cbx.size(3)], self.spp_num_outputs)
        x = x_Cbx.squeeze(1)
        """
        res_Cbx = res_Cbx.squeeze()
        x = self.afc1(res_Cbx)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.afc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.afc3(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.afc4(x)
        x = self.ln4(x)
        x = self.drop(x)
        mu = torch.sigmoid(self.mu(x))
        return mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'action': 4, 'num_inputs': 4,
        'num_output': 4}]
