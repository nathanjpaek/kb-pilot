import torch


class PDController(torch.nn.Module):

    def __init__(self):
        super(PDController, self).__init__()

    def forward(self, kp, kd, position, velocity, des_position, des_velocity):
        return kp * (des_position - position) + kd * (des_velocity - velocity)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]),
        torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
