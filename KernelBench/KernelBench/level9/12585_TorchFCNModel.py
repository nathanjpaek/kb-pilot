import torch


class TorchFCNModel(torch.nn.Module):

    def __init__(self, inputD, outputD, hiddenC=2, hiddenD=36):
        super(TorchFCNModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else
            'cpu')
        self.inputD, self.outputD = inputD, outputD
        self.hiddenC, self.hiddenD = hiddenC, hiddenD
        self.linearBegin = torch.nn.Linear(inputD, hiddenD)
        self.linearHidden1 = torch.nn.Linear(hiddenD, hiddenD)
        self.linearHidden2 = torch.nn.Linear(hiddenD, hiddenD)
        self.linearOut = torch.nn.Linear(hiddenD, outputD)

    def forward(self, x):
        h_relu = self.linearBegin(x).clamp(min=0)
        h_relu1 = self.linearHidden1(h_relu).clamp(min=0)
        h_relu2 = self.linearHidden2(h_relu1).clamp(min=0)
        y_pred = self.linearOut(h_relu2)
        return y_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputD': 4, 'outputD': 4}]
