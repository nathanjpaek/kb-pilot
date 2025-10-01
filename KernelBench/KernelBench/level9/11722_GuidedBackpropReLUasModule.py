from torch.autograd import Function
import torch
import torch.cuda


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(
            input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, _output = self.saved_tensors
        grad_input = None
        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(
            input_img), torch.addcmul(torch.zeros(input_img.size()).type_as
            (input_img), grad_output, positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUasModule(torch.nn.Module):

    def __init__(self):
        super(GuidedBackpropReLUasModule, self).__init__()

    def forward(self, input_img):
        return GuidedBackpropReLU.apply(input_img)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
