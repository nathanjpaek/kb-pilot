import torch


class EmbeddingModel(torch.nn.Module):

    @staticmethod
    def forward(inputs):
        return inputs.repeat(1, 10)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
