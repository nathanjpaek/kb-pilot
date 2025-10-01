import torch


class DropConnect(torch.nn.Module):

    def __init__(self, p):
        super(DropConnect, self).__init__()
        self.p = p

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs.shape[2]
        inputs.shape[3]
        channel_size = inputs.shape[1]
        keep_prob = 1 - self.p
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, channel_size, 1, 1], dtype
            =inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p': 4}]
