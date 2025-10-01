import torch
import torch.nn as nn
import torch.fft


class VGGLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)
        input_features = x
        output_features = y
        for idx, (input_feature, output_feature) in enumerate(zip(
            input_features, output_features)):
            loss += self.mse_loss(output_feature, input_feature).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
