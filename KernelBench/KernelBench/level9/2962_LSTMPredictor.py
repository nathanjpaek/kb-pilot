import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):

    def __init__(self, look_back, target_days):
        super(LSTMPredictor, self).__init__()
        self.layer_a = nn.Linear(look_back, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, target_days)

    def predict(self, input):
        with torch.no_grad():
            return self.forward(input).tolist()

    def forward(self, input):
        logits = self.output(self.relu(self.layer_a(input)))
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'look_back': 4, 'target_days': 4}]
