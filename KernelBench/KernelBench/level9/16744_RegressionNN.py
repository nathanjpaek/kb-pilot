import torch
import numpy as np
import torch.nn as nn


class RegressionNN(nn.Module):

    def __init__(self, feature_number):
        super(RegressionNN, self).__init__()
        self.feature_number = feature_number
        self.fc1 = nn.Linear(self.feature_number, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x

    def predict(self, X_test):
        X_test = np.array(X_test)
        X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
        y_pred = self.forward(X_test)
        y_pred = y_pred.detach().numpy()
        return y_pred.flatten()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_number': 4}]
