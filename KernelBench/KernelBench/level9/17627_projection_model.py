import torch


class projection_model(torch.nn.Module):

    def __init__(self, neo_hidden, clip_hidden=512):
        super(projection_model, self).__init__()
        self.fc1 = torch.nn.Linear(neo_hidden, neo_hidden // 2)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(neo_hidden // 2, clip_hidden)

    def forward(self, input_tensor):
        out = self.act(self.fc1(input_tensor))
        return self.fc2(out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'neo_hidden': 4}]
