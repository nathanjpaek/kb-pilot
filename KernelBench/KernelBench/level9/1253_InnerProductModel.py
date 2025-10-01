import torch


class InnerProductModel(torch.nn.Module):

    @staticmethod
    def is_valid_model_type(model_type):
        raise NotImplementedError

    @staticmethod
    def get_model_from_type(model_type):
        raise NotImplementedError

    @property
    def loss_criterion(self):
        return torch.nn.MSELoss()

    def __init__(self, n):
        super().__init__()
        self.layer = torch.nn.Linear(n, 1, bias=False)
        self.layer.weight.data = torch.arange(n, dtype=torch.float32)

    def forward(self, x):
        return self.layer(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4}]
