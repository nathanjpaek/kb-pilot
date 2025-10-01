from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class DNN(nn.Module):

    def __init__(self, config):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(784, int(config['hidden_layer1']))
        self.dropout = nn.Dropout2d(float(config['drop_out']))
        self.fc2 = nn.Linear(int(config['hidden_layer1']), int(config[
            'hidden_layer2']))
        self.fc = nn.Linear(int(config['hidden_layer2']), 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_layer1=1, drop_out=0.5,
        hidden_layer2=1)}]
