import torch


class DeepNeuralNet(torch.nn.Module):
    """
         This is a six-layer neural network.
         This is the default network for initializing sigma and center parameters
    """

    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3,
        n_hidden4, n_hidden5, n_hidden6, n_output):
        """
                 Initialization
                 :param n_feature: Feature number
                 :param n_hidden: the number of hidden layer neurons
                 :param n_output: output number
        """
        super(DeepNeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_feature, n_hidden1)
        self.fc2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.fc4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.fc5 = torch.nn.Linear(n_hidden4, n_hidden5)
        self.fc6 = torch.nn.Linear(n_hidden5, n_hidden6)
        self.predict = torch.nn.Linear(n_hidden6, n_output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.predict(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_feature': 4, 'n_hidden1': 4, 'n_hidden2': 4,
        'n_hidden3': 4, 'n_hidden4': 4, 'n_hidden5': 4, 'n_hidden6': 4,
        'n_output': 4}]
