from _paritybench_helpers import _mock_config
import torch


class Net(torch.nn.Module):

    def __init__(self, configs):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(configs['input_size'], configs[
            'hidden_size'])
        self.fc1_activate = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(configs['hidden_size'], configs[
            'output_size'])
        self.out_activate = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_activate(x)
        x = self.fc2(x)
        out = self.out_activate(x)
        return out

    def initialize_weights(self):
        for m in self.modules():
            torch.nn.init.normal_(m.weight.data, 0.01)
            torch.nn.init.constant_(m.bias.data, 0.01)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'configs': _mock_config(input_size=4, hidden_size=4,
        output_size=4)}]
