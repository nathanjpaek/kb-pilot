import torch
import torch.nn as nn
import torch.fft


class GramMatrix(torch.nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        gram_matrix.div_(h * w)
        return gram_matrix


class GramLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.gram_matrix = GramMatrix()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)
        input_features = x
        output_features = y
        for idx, (input_feature, output_feature) in enumerate(zip(
            input_features, output_features)):
            gram_out = self.gram_matrix(output_feature)
            gram_in = self.gram_matrix(input_feature)
            loss += self.l1_loss(gram_out, gram_in).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
