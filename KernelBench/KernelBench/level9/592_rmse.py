import torch


class rmse(torch.nn.Module):

    def __init__(self):
        super(rmse, self).__init__()

    def forward(self, y_true, y_pred):
        mse = torch.mean((y_pred - y_true) ** 2, axis=-1)
        rmse = torch.sqrt(mse + 1e-07)
        return torch.mean(rmse)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
