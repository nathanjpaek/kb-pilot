import torch
import torch.nn as nn


class Normalize(nn.Module):
    """ 
    Scale Audio to be between -1 and 1
    """

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, audio: 'torch.Tensor'):
        if len(audio.shape) != 2:
            raise ValueError('Audio should be 2D: [batch_size X audio_length]')
        if audio.shape[1] < 1:
            raise ValueError('Audio length is zero')
        max_value = torch.max(audio, dim=1)[0].detach()
        min_value = torch.min(audio, dim=1)[0].detach()
        max_value = torch.unsqueeze(max_value, 1)
        min_value = torch.unsqueeze(min_value, 1)
        audio = (audio - min_value) / (max_value - min_value + 1e-10)
        return audio * 2 - 1


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
