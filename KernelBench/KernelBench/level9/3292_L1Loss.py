import torch


class L1Loss(torch.nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, cleaned_images, images):
        return self.loss(cleaned_images, images)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
