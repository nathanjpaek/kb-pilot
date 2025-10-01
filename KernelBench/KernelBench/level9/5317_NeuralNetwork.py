import torch


class NeuralNetwork(torch.nn.Module):
    """
    Neural network class of fully connected layers

    Args:
        n_input_feature : int
            number of input features
        n_output : int
            number of output classes
    """

    def __init__(self, n_input_feature, n_output):
        super(NeuralNetwork, self).__init__()
        self.fully_connected_L1 = torch.nn.Linear(n_input_feature, 512)
        self.fully_connected_L2 = torch.nn.Linear(512, 256)
        self.fully_connected_L3 = torch.nn.Linear(256, 128)
        self.output = torch.nn.Linear(128, n_output)

    def forward(self, x):
        """
            Args:
                x : torch.Tensor
                    tensor of N x d

            Returns:
                torch.Tensor
                    tensor of n_output
        """
        x_1 = self.fully_connected_L1(x)
        eta_x1 = torch.nn.functional.relu(x_1)
        x_2 = self.fully_connected_L2(eta_x1)
        eta_x2 = torch.nn.functional.relu(x_2)
        x_3 = self.fully_connected_L3(eta_x2)
        eta_x3 = torch.nn.functional.relu(x_3)
        output = self.output(eta_x3)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input_feature': 4, 'n_output': 4}]
