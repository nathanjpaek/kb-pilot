import torch
import torch.nn
import torch.onnx
import torch.utils.checkpoint


class NeuralNetNonDifferentiableOutput(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetNonDifferentiableOutput, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out1 = self.relu(out)
        out2 = self.fc2(out1)
        mask1 = torch.gt(out1, 0.01)
        mask1 = mask1.long()
        mask2 = torch.lt(out2, 0.02)
        mask2 = mask2.long()
        return out1, mask1, out2, mask2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'num_classes': 4}]
