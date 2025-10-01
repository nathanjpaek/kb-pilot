import torch


def sequence_length_3D(sequence: 'torch.Tensor') ->torch.Tensor:
    used = torch.sign(torch.amax(torch.abs(sequence), dim=2))
    length = torch.sum(used, 1)
    length = length.int()
    return length


class ReduceLast(torch.nn.Module):

    def forward(self, inputs, mask=None):
        batch_size = inputs.shape[0]
        sequence_length = sequence_length_3D(inputs) - 1
        sequence_length[sequence_length < 0] = 0
        gathered = inputs[torch.arange(batch_size), sequence_length.type(
            torch.int64)]
        return gathered


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
