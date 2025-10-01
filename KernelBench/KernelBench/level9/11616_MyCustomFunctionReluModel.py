import torch
import torch.nn
import torch.onnx


class MyCustomFunctionReluModel(torch.nn.Module):

    def __init__(self):
        super().__init__()


        class MyReLU(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input
        self.relu = MyReLU.apply

    def forward(self, input):
        return self.relu(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
