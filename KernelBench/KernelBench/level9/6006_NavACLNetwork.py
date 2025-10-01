import torch
import torch.nn as nn


class NavACLNetwork(nn.Module):

    def __init__(self, task_param_dim, hidden_dim, init_w=0.0005):
        super(NavACLNetwork, self).__init__()
        self.layer_1 = nn.Linear(task_param_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, 1)
        nn.init.kaiming_uniform_(self.layer_1.weight, mode='fan_in',
            nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer_2.weight, mode='fan_in',
            nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer_3.weight, mode='fan_in',
            nonlinearity='relu')
        self.m1 = torch.nn.LeakyReLU(0.1)
        self.m2 = torch.nn.LeakyReLU(0.1)
        self.m3 = torch.nn.LeakyReLU(0.1)

    def forward(self, inputs):
        x = self.m1(self.layer_1(inputs))
        x = self.m2(self.layer_2(x))
        x = self.m3(self.layer_3(x))
        x = torch.sigmoid(self.layer_out(x))
        return x

    def get_task_success_prob(self, task_param_array):
        torch_task_param_array = torch.FloatTensor(task_param_array)
        difficulty_estimate = self.forward(torch_task_param_array)
        return difficulty_estimate.detach().cpu().numpy()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'task_param_dim': 4, 'hidden_dim': 4}]
