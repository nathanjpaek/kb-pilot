import torch
import numpy as np
from torch import as_tensor
from torch import no_grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AsModelNet(nn.Module):

    @staticmethod
    def chunk_it(xx):
        d = []
        for x in xx:
            d.append(x)
            if len(d) >= 3:
                yield as_tensor(d)
                d = []
        if d:
            yield as_tensor(d)

    def fit(self, xx, yy):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        chunk_x = (as_tensor(x.reshape(1, *self.shape).astype(np.float32)) for
            x in xx)
        chunk_y = (as_tensor(y.reshape(1)) for y in yy)
        for epoch in range(5):
            for x, y in zip(chunk_x, chunk_y):
                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

    def predict(self, xx):
        with no_grad():
            tensor = self(as_tensor(xx.reshape(-1, *self.shape).astype(np.
                float32)))
            return tensor.numpy().argmax(axis=1)

    def eye_probability(self, vector):
        vector = vector.reshape(1, -1)
        with no_grad():
            tensor = self(as_tensor(vector.reshape(-1, *self.shape).astype(
                np.float32)))
        return tensor[0, 1].item()


class Net2(AsModelNet):

    def __init__(self, shape_size):
        super().__init__()
        mid_size = round(np.sqrt(shape_size))
        self.shape = shape_size,
        self.fc1 = nn.Linear(shape_size, mid_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(mid_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'shape_size': 4}]
