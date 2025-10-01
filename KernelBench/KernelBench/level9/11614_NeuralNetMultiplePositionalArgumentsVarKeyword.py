import torch
import torch.nn
import torch.onnx


class NeuralNetMultiplePositionalArgumentsVarKeyword(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArgumentsVarKeyword, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2, **kwargs):
        model_input = input1 + input2
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'num_classes': 4}]
