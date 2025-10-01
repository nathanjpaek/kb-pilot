import torch
import torch.nn
import torch.onnx


class NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency(torch.
    nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMultiplePositionalArgumentsMultiOutputsWithDependency,
            self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2):
        model_input = input1 + input2
        out1 = self.fc1(model_input)
        out1 = self.softmax(out1)
        out2 = self.fc2(out1)
        return out1, out2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'num_classes': 4}]
