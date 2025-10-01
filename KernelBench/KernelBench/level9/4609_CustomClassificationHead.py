from _paritybench_helpers import _mock_config
import torch
from torch import nn


class CustomClassificationHead(nn.Module):

    def __init__(self, config, input_dim, n_labels):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, n_labels)
        self.dropout = nn.Dropout(p=0.3)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x[:, 0, :]
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(self.dropout(x)))
        x = self.prelu3(self.fc3(self.dropout(x)))
        return self.fc4(self.dropout(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(), 'input_dim': 4, 'n_labels': 4}]
