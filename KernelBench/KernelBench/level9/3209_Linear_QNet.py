import torch
import torch.nn as nn


class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.leakyrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)
        self.sigmaoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.leakyrelu(x)
        x = self.linear2(x)
        x = self.leakyrelu(x)
        x = self.linear3(x)
        x = self.sigmaoid(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size_1': 4, 'hidden_size_2': 4,
        'output_size': 4}]
