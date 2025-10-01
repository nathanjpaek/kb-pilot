import torch
import torch.nn as nn
import torch.autograd
import torch.backends.cudnn


class LinearModel(nn.Module):
    """
    NetModel class for the neural network. inherits from NetModel.
    """

    def __init__(self, input_size, output_size, hidden_size):
        """
        Initialize the model.
        :param input_size:
        :param output_size:
        """
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 50)
        self.fc4 = nn.PReLU()
        self.fc5 = nn.Linear(50, output_size)
        self.out = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.
        :param x:
        :return: logits
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return self.out(x)

    def save(self, filename: 'str'):
        """
        Save the model to a file.
        :param filename:
        :return: None
        """
        path = Path(filename)
        with click_spinner.spinner('Saving model to {}'.format(path)):
            with path.open('wb') as f:
                torch.save(self, f)
        typer.secho(f'{self.__class__.__name__} saved', fg='green')
        return None

    @classmethod
    def load(cls, path):
        """
        Load a model from a file.
        :param path: path to the model file (str)
        :return: NetModel instance
        """
        path = Path(path)
        with click_spinner.spinner('Loading model from {}'.format(path)):
            with path.open('rb') as f:
                model = torch.load(f)
        name = model.__class__.__name__
        typer.secho(f'{name} loaded', fg='green')
        return model


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'hidden_size': 4}]
